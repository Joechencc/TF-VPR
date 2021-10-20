from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import math
import random
from .resnet_mod import *

from matplotlib import pyplot as plt
import os

def draw_diagram(pcs, rot_pcs, inside_flag=True, output_trusted_path = "./results/visualization_2"):
    for count, pc in enumerate(pcs.squeeze()):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        pc = pc.cpu().detach().numpy()
        ax.scatter(pc[:,0], pc[:,1])
        if inside_flag:
            plt.savefig(os.path.join(output_trusted_path, "pcl_"+str(count)+".jpg"))
        else:
            plt.savefig(os.path.join(output_trusted_path, "compare_pcl_"+str(count)+".jpg"))

    for count, rot_pc in enumerate(rot_pcs.squeeze()):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        rot_pc = rot_pc.cpu().detach().numpy()
        ax.scatter(rot_pc[:,0], rot_pc[:,1])
        if inside_flag:
            plt.savefig(os.path.join(output_trusted_path, "rot_pcl_"+str(count)+".jpg"))
        else:
            plt.savefig(os.path.join(output_trusted_path, "rot_compare_pcl_"+str(count)+".jpg"))

def rotate_point_cloud_N3(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
    rotation is per shape based along up direction
    Input:
    Nx3 array, original batch of point clouds
    Return:
    Nx3 array, rotated batch of point clouds
    """
    # rotated_data = torch.zeros(batch_data.shape, dtype=torch.float32, device=batch_data.device)
    rotation_angle = (np.random.uniform()*2*np.pi) - np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = torch.tensor([[cosval, -sinval,0],
                               [sinval, cosval,0],
                               [0, 0,1]], dtype=torch.float32, device=batch_data.device)
    
    # rotation_matrix.requires_grad = True
    # for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        #-90 to 90
    rotated_data= torch.matmul(
            batch_data, rotation_matrix).detach ()
            
    return rotated_data
    
class NetVLADLoupe(nn.Module):
    def __init__(self, feature_size, max_samples, cluster_size, output_dim,
                 gating=True, add_batch_norm=True, is_training=True):
        super(NetVLADLoupe, self).__init__()
        self.feature_size = feature_size
        self.max_samples = max_samples
        self.output_dim = output_dim
        self.is_training = is_training
        self.gating = gating
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size
        self.softmax = nn.Softmax(dim=-1)
        self.cluster_weights = nn.Parameter(torch.randn(
            feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.cluster_weights2 = nn.Parameter(torch.randn(
            1, feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.hidden1_weights = nn.Parameter(
            torch.randn(cluster_size * feature_size, output_dim) * 1 / math.sqrt(feature_size))

        if add_batch_norm:
            self.cluster_biases = None
            self.bn1 = nn.BatchNorm1d(cluster_size)
        else:
            self.cluster_biases = nn.Parameter(torch.randn(
                cluster_size) * 1 / math.sqrt(feature_size))
            self.bn1 = None

        self.bn2 = nn.BatchNorm1d(output_dim)

        if gating:
            self.context_gating = GatingContext(
                output_dim, add_batch_norm=add_batch_norm)

    def forward(self, x):
        x = x.transpose(1, 3).contiguous()
        x = x.view((-1, self.max_samples, self.feature_size))
        activation = torch.matmul(x, self.cluster_weights)
        if self.add_batch_norm:
            # activation = activation.transpose(1,2).contiguous()
            activation = activation.view(-1, self.cluster_size)
            activation = self.bn1(activation)
            activation = activation.view(-1,
                                         self.max_samples, self.cluster_size)
            # activation = activation.transpose(1,2).contiguous()
        else:
            activation = activation + self.cluster_biases
        activation = self.softmax(activation)
        activation = activation.view((-1, self.max_samples, self.cluster_size))

        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weights2

        activation = torch.transpose(activation, 2, 1)
        x = x.view((-1, self.max_samples, self.feature_size))
        vlad = torch.matmul(activation, x)
        vlad = torch.transpose(vlad, 2, 1)
        vlad = vlad - a

        vlad = F.normalize(vlad, dim=1, p=2)
        vlad = vlad.view((-1, self.cluster_size * self.feature_size))
        vlad = F.normalize(vlad, dim=1, p=2)

        vlad = torch.matmul(vlad, self.hidden1_weights)

        vlad = self.bn2(vlad)

        if self.gating:
            vlad = self.context_gating(vlad)

        return vlad


class GatingContext(nn.Module):
    def __init__(self, dim, add_batch_norm=True):
        super(GatingContext, self).__init__()
        self.dim = dim
        self.add_batch_norm = add_batch_norm
        self.gating_weights = nn.Parameter(
            torch.randn(dim, dim) * 1 / math.sqrt(dim))
        self.sigmoid = nn.Sigmoid()

        if add_batch_norm:
            self.gating_biases = None
            self.bn1 = nn.BatchNorm1d(dim)
        else:
            self.gating_biases = nn.Parameter(
                torch.randn(dim) * 1 / math.sqrt(dim))
            self.bn1 = None

    def forward(self, x):
        gates = torch.matmul(x, self.gating_weights)

        if self.add_batch_norm:
            gates = self.bn1(gates)
        else:
            gates = gates + self.gating_biases

        gates = self.sigmoid(gates)

        activation = x * gates

        return activation


class Flatten(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, input):
        return input.view(input.size(0), -1)


class STN2d(nn.Module):
    def __init__(self, num_points=256, k=2, use_bn=True):
        super(STN2d, self).__init__()
        self.k = k
        self.kernel_size = 2 if k == 2 else 1
        self.channels = 1 if k == 2 else k
        self.num_points = num_points
        self.use_bn = use_bn
        self.conv1 = torch.nn.Conv2d(self.channels, 64, (1, self.kernel_size))
        self.conv2 = torch.nn.Conv2d(64, 128, (1,1))
        self.conv3 = torch.nn.Conv2d(128, 1024, (1,1))
        self.mp1 = torch.nn.MaxPool2d((num_points, 1), 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.zero_()
        self.relu = nn.ReLU()

        if use_bn:
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(1024)
            self.bn4 = nn.BatchNorm1d(512)            
            self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        if self.use_bn:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
        x = self.mp1(x)    
        x = x.view(-1, 1024)

        if self.use_bn:
            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        
        iden = Variable(torch.from_numpy(np.eye(self.k).astype(np.float32))).view(
               1, self.k*self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class STN3d(nn.Module):
    def __init__(self, num_points=2500, k=3, use_bn=True):
        super(STN3d, self).__init__()
        self.k = k
        self.kernel_size = 3 if k == 3 else 1
        self.channels = 1 if k == 3 else k
        self.num_points = num_points
        self.use_bn = use_bn
        self.conv1 = torch.nn.Conv2d(self.channels, 64, (1, self.kernel_size))
        self.conv2 = torch.nn.Conv2d(64, 128, (1,1))
        self.conv3 = torch.nn.Conv2d(128, 1024, (1,1))
        self.mp1 = torch.nn.MaxPool2d((num_points, 1), 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.zero_()
        self.relu = nn.ReLU()

        if use_bn:
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(1024)
            self.bn4 = nn.BatchNorm1d(512)
            self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        if self.use_bn:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        if self.use_bn:
            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).astype(np.float32))).view(
            1, self.k*self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class ObsFeatAVD(nn.Module):
    """Feature extractor for 2D organized point clouds"""
    def __init__(self, n_out=1024, num_points=2500, global_feat=True, feature_transform=False, max_pool=True):
        super(ObsFeatAVD, self).__init__()
        self.n_out = n_out
        self.global_feature = global_feat
        self.feature_transform = feature_transform
        self.max_pool = max_pool
        k = 3
        p = int(np.floor(k / 2)) + 2
        self.stn = STN3d(num_points=num_points, k=3, use_bn=False)
        self.feature_trans = STN3d(num_points=num_points, k=64, use_bn=False)
        self.conv1 = nn.Conv2d(3,64,kernel_size=k,padding=p,dilation=3)        
        self.conv2 = nn.Conv2d(64,128,kernel_size=k,padding=p,dilation=3)
        self.conv3 = nn.Conv2d(128,256,kernel_size=k,padding=p,dilation=3)
        self.conv7 = nn.Conv2d(256,self.n_out,kernel_size=k,padding=p,dilation=3)
        self.amp = nn.AdaptiveMaxPool2d(1)
   
    def forward(self, x):
        batchsize = x.size()[0]
        trans = self.stn(x)
        x = torch.matmul(torch.squeeze(x), trans)
        x = x.view(batchsize, 1, -1, 3)
        x = x.permute(0,3,1,2)
        assert(x.shape[1]==3),"the input size must be <Bx3xHxW> "
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv7(x)
        if self.max_pool:
            x = self.amp(x) 
        #x = x.view(-1,self.n_out) #<Bxn_out>
        x = x.permute(0,2,3,1)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, num_points=256, global_feat=True, feature_transform=False, max_pool=True):
        super(PointNetfeat, self).__init__()
        self.stn = STN2d(num_points=num_points, k=2, use_bn=False)
        self.feature_trans = STN2d(num_points=num_points, k=64, use_bn=False)
        self.apply_feature_trans = feature_transform
        self.conv1 = torch.nn.Conv1d(1, 64, (1, 2))
        self.conv2 = torch.nn.Conv1d(64, 64, (1, 1))
        self.conv3 = torch.nn.Conv1d(64, 64, (1, 1))
        self.conv4 = torch.nn.Conv1d(64, 128, (1, 1))
        self.conv5 = torch.nn.Conv2d(128, 1024, (1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(1024)
        self.mp1 = torch.nn.MaxPool2d((num_points, 1), 1)
        self.num_points = num_points
        self.global_feat = global_feat
        self.max_pool = max_pool

    def forward(self, x):
        x = x[:,:,:,:2]
        batchsize = x.size()[0]
        trans = self.stn(x)
        x = torch.matmul(torch.squeeze(x), trans)
        x = x.view(batchsize, 1, -1, 2)
        #x = x.transpose(2,1)
        #x = torch.bmm(x, trans)
        #x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        pointfeat = x
        #print("x_before_trans:"+str(x))
        if self.apply_feature_trans:
            f_trans = self.feature_trans(x)
            x = torch.squeeze(x)
            if batchsize == 1:
                x = torch.unsqueeze(x, 0)
            x = torch.matmul(x.transpose(1, 2), f_trans)
            x = x.transpose(1, 2).contiguous()
            x = x.view(batchsize, 64, -1, 1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))
        if not self.max_pool:
            return x
        else:
            x = self.mp1(x)
            x = x.view(-1, 1024)
            if self.global_feat:
                return x, trans
            else:
                x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans


class PointNetfeatCNN(nn.Module):
    def __init__(self, num_points=256, global_feat=True, feature_transform=False, max_pool=True):
        super(PointNetfeatCNN, self).__init__()
        self.stn = STN2d(num_points=num_points, k=2, use_bn=False)
        self.feature_trans = STN2d(num_points=num_points, k=64, use_bn=False)
        self.apply_feature_trans = feature_transform
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(3,1), stride=1, padding=(2,0),
                                            bias=False, padding_mode='circular')
        
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=(3,2), stride=1, padding=(2,1),
                                                            bias=False, padding_mode='circular')
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=(3,1), stride=1, padding=(2,0),
                                                            bias=False, padding_mode='circular')
        self.conv4 = torch.nn.Conv2d(64, 128, kernel_size=(3,1), stride=1, padding=(2,0),
                                                            bias=False, padding_mode='circular')
        self.conv5 = torch.nn.Conv2d(128, 1024, kernel_size=(3,1), stride=1, padding=(2,0),
                                                            bias=False, padding_mode='circular')
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(1024)
        self.mp1 = torch.nn.MaxPool2d((num_points, 1), 1)
        self.num_points = num_points
        self.global_feat = global_feat
        self.max_pool = max_pool

    def process_data(self,ob,low_th = -20,high_th = 20):
        for i in range(ob.size()[0]):
            seed = random.randint(low_th, high_th)
            ob[i] = torch.roll(ob[i], seed, dims=1)
            return ob

    def forward(self, x):
        rot_x = rotate_point_cloud_N3(x)[:,:,:,:2]
        # print("rot_x:"+str(rot_x.shape))  
        x = x[:,:,:,:2]
        # print("x:"+str(x.shape))

        # draw_diagram(x, rot_x)
        # assert(0)

        batchsize = x.size()[0]
        threshold = batchsize//2
        x = self.process_data(x, low_th=-threshold, high_th=threshold)
        rot_x = self.process_data(rot_x, low_th=-threshold, high_th=threshold)
        
        '''
        trans = self.stn(x)
        x = torch.matmul(torch.squeeze(x), trans)
        '''
        x = x.view(batchsize, 1, -1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        pointfeat = x
        
        if self.apply_feature_trans:
            f_trans = self.feature_trans(x)
            x = torch.squeeze(x)
            if batchsize == 1:
                x = torch.unsqueeze(x, 0)
            x = torch.matmul(x.transpose(1, 2), f_trans)
            x = x.transpose(1, 2).contiguous()
            x = x.view(batchsize, 64, -1, 1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = self.bn5(self.conv5(x))

        '----------------------------------'

        rot_x = rot_x.view(batchsize, 1, -1, 2)
        rot_x = F.relu(self.bn1(self.conv1(rot_x)))
        rot_x = F.relu(self.bn2(self.conv2(rot_x)))
        pointfeat_rot_x = rot_x
        
        if self.apply_feature_trans:
            f_trans_rot = self.feature_trans(rot_x)
            rot_x = torch.squeeze(rot_x)
            if batchsize == 1:
                rot_x = torch.unsqueeze(rot_x, 0)
            rot_x = torch.matmul(rot_x.transpose(1, 2), f_trans_rot)
            rot_x = rot_x.transpose(1, 2).contiguous()
            rot_x = rot_x.view(batchsize, 64, -1, 1)
        rot_x = F.relu(self.bn3(self.conv3(rot_x)))
        rot_x = F.relu(self.bn4(self.conv4(rot_x)))
        
        rot_x = self.bn5(self.conv5(rot_x))

        '----------------------------------'

        if not self.max_pool:
            return x, rot_x
        else:
            x = self.mp1(x)
            x = x.view(-1, 1024)

            rot_x = self.mp1(rot_x)
            rot_x = rot_x.view(-1, 1024)
            
            if self.global_feat:
                return x, trans, rot_x, trans_rot
            else:
                x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
                rot_x = rot_x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans, torch.cat([rot_x, pointfeat_rot], 1), trans_rot

        # if not self.max_pool:
        #     return x
        # else:
        #     x = self.mp1(x)
        #     x = x.view(-1, 1024)
            
        #     if self.global_feat:
        #         return x, trans
        #     else:
        #         x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
        #         return torch.cat([x, pointfeat], 1), trans

class PointNetVlad(nn.Module):
    def __init__(self, num_points=256, global_feat=True, feature_transform=False, max_pool=True, output_dim=1024):
        super(PointNetVlad, self).__init__()
        #self.point_net = PointNetfeat(num_points=num_points, global_feat=global_feat,
        #                              feature_transform=feature_transform, max_pool=max_pool)
        #self.obs_feat_extractor = ObsFeatAVD(n_out=1024, num_points=num_points, global_feat=global_feat,
        #                              feature_transform=feature_transform, max_pool=max_pool)
        self.obs_feat_extractor = PointNetfeatCNN(num_points=num_points, global_feat=global_feat,
                                               feature_transform=feature_transform, max_pool=max_pool)
        self.net_vlad = NetVLADLoupe(feature_size=1024, max_samples=num_points, cluster_size=64,
                                     output_dim=output_dim, gating=True, add_batch_norm=True,
                                     is_training=True)

    def forward(self, x):
        #x = self.point_net(x)
        x, rot_x = self.obs_feat_extractor(x)
        x = self.net_vlad(x)
        rot_x = self.net_vlad(rot_x)

        return x, rot_x


if __name__ == '__main__':
    num_points = 256
    sim_data = Variable(torch.rand(44, 1, num_points, 3))
    sim_data = sim_data.cuda()

    pnv = PointNetVlad.PointNetVlad(global_feat=True, feature_transform=True, max_pool=False,
                                    output_dim=256, num_points=num_points).cuda()
    pnv.train()
    out3 = pnv(sim_data)
    print('pnv', out3.size())
