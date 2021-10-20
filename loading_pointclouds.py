import os
import pickle5 as pickle
import numpy as np
import random
import config as cfg
import open3d as o3d
from open3d import read_point_cloud
from matplotlib import pyplot as plt

def get_queries_dict(filename):
    # key:{'query':file,'positives':[files],'negatives:[files], 'neighbors':[keys]}
    with open(filename, 'rb') as handle:
        print("filename:"+str(filename))
        queries = pickle.load(handle)
        print("Queries Loaded.")
        return queries


def get_sets_dict(filename):
    #[key_dataset:{key_pointcloud:{'query':file,'northing':value,'easting':value}},key_dataset:{key_pointcloud:{'query':file,'northing':value,'easting':value}}, ...}
    with open(filename, 'rb') as handle:
        trajectories = pickle.load(handle)
        print("Trajectories Loaded.")
        return trajectories


def load_pc_file(filename,full_path=False):
    # returns Nx3 matrix
    #print("filename:"+str(filename))
    if full_path:
        pc = o3d.read_point_cloud(os.path.join(filename))
    else:
        pc = o3d.read_point_cloud(os.path.join("/mnt/ab0fe826-9b3c-455c-bb72-5999d52034e0/deepmapping/benchmark_datasets/", filename))
    pc = np.asarray(pc.points, dtype=np.float32)
    
    if(pc.shape[0] != 256):
        print("Error in pointcloud shape")
        return np.array([])

    #pc = np.reshape(pc,(pc.shape[0]//3, 3))
    return pc


def load_pc_files(filenames,full_path):
    pcs = []
    for filename in filenames:
        # print(filename)
        pc = load_pc_file(filename,full_path=full_path)
        if(pc.shape[0] != 256):
            continue
        pcs.append(pc)
    pcs = np.array(pcs)
    return pcs

def save_fig(pc, i_file_num, inside_flag=True, output_trusted_path = "/home/cc/2_Unsupervised-PointNetVlad_aug_devel_2/results/visualization_2/"):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.scatter(pc[:,0], pc[:,1])
    if inside_flag:
        plt.savefig(os.path.join(output_trusted_path, "pcl_"+str(i_file_num)+".jpg"))
    else:
        plt.savefig(os.path.join(output_trusted_path, "compare_pcl_"+str(i_file_num)+".jpg"))

def load_pos_neg_pc_files(filenames,full_path):
    pcs = []
    for filename in filenames:
        # print(filename)
        pc = load_pc_file(filename,full_path=full_path)
        if(pc.shape[0] != 256):
            continue
        
        for i in range(2):
            #save_fig(pc,i,inside_flag=True)
            rotated_pcl = rotate_point_cloud_N3(pc)
            #save_fig(rotated_pcl,i,inside_flag=False)
            pcs.append(rotated_pcl)
    
    pcs = np.array(pcs)
    return pcs

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
    rotation is per shape based along up direction
    Input:
    BxNx2 array, original batch of point clouds
    Return:
    BxNx2 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    rotation_angle = (np.random.uniform()*2*np.pi) - np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, -sinval], 
                                [sinval, cosval]])
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        #-90 to 90
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(
                               shape_pc, rotation_matrix) 
    return rotated_data

def rotate_point_cloud_N3(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
    rotation is per shape based along up direction
    Input:
    Nx3 array, original batch of point clouds
    Return:
    Nx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    rotation_angle = (np.random.uniform()*2*np.pi) - np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, -sinval,0],
                               [sinval, cosval,0],
                               [0, 0,1]])
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        #-90 to 90
        shape_pc = batch_data
        rotated_data= np.dot(
                shape_pc, rotation_matrix)
    return rotated_data

def jitter_point_cloud(batch_data, sigma=0.005, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def get_query_tuple(dict_value, num_pos, num_neg, QUERY_DICT, hard_neg=[], other_neg=False):
        # get query tuple for dictionary entry
        # return list [query,positives,negatives]

    query = load_pc_file(dict_value["query"],True)  # Nx3

    random.shuffle(dict_value["positives"])
    pos_files = []

    for i in range(num_pos):
        pos_files.append(QUERY_DICT[dict_value["positives"][i]]["query"])
    #positives= load_pc_files(dict_value["positives"][0:num_pos])
    positives = load_pos_neg_pc_files(pos_files,full_path=True)
    '''
    B, P, _ = positives.shape
    new_positives = np.zeros((B*(cfg.ROT_NUM+1),P,3), dtype = positives.dtype)
    for pb in range(positives.shape[0]):
        positve_pcl = positives[pb][:,:2]
        new_positives[pb*(cfg.ROT_NUM+1),:,:2] = positve_pcl
        for r_n in range(cfg.ROT_NUM):
            rotated_positve_pcl = rotate_point_cloud(positve_pcl)
            #print("rotated_positve_pcl:"+str(rotated_positve_pcl))
            new_positives[pb*(cfg.ROT_NUM+1)+r_n+1,:,:2] = rotated_positve_pcl
    new_positives = np.asarray(new_positives)
    positives = new_positives
    '''
    neg_files = []
    neg_indices = []
    if(len(hard_neg) == 0):
        random.shuffle(dict_value["negatives"])
        for i in range(num_neg):
            neg_files.append(QUERY_DICT[dict_value["negatives"][i]]["query"])
            neg_indices.append(dict_value["negatives"][i])
        
    else:
        random.shuffle(dict_value["negatives"])
        for count, i in enumerate(hard_neg):
            neg_files.append(QUERY_DICT[i]["query"])
            neg_indices.append(i)
        j = 0
        while(len(neg_files) < num_neg):

            if not dict_value["negatives"][j] in hard_neg:
                neg_files.append(
                    QUERY_DICT[dict_value["negatives"][j]]["query"])
                neg_indices.append(dict_value["negatives"][j])
            j += 1

    negatives = load_pc_files(neg_files,full_path=True)
    if other_neg is False:
        return [query, positives, negatives]
    # For Quadruplet Loss
    else:
        # get neighbors of negatives and query
        neighbors = []
        for pos in dict_value["positives"]:
            neighbors.append(pos)
        for neg in neg_indices:
            for pos in QUERY_DICT[neg]["positives"]:
                neighbors.append(pos)
        possible_negs = list(set(QUERY_DICT.keys())-set(neighbors))
        random.shuffle(possible_negs)

        if(len(possible_negs) == 0):
            return [query, positives, negatives, np.array([])]

        neg2 = load_pc_file(QUERY_DICT[possible_negs[0]]["query"],full_path=True)
        return [query, positives, negatives, neg2]


def get_rotated_tuple(dict_value, num_pos, num_neg, QUERY_DICT, hard_neg=[], other_neg=False):
    query = load_pc_file(dict_value["query"],True)  # Nx3
    q_rot = rotate_point_cloud(np.expand_dims(query, axis=0))
    q_rot = np.squeeze(q_rot)

    random.shuffle(dict_value["positives"])
    pos_files = []
    for i in range(num_pos):
        pos_files.append(QUERY_DICT[dict_value["positives"][i]]["query"])
    #positives= load_pc_files(dict_value["positives"][0:num_pos])
    #print("pos_files:"+str(pos_files))
    #assert(0)
    positives = load_pc_files(pos_files,True)
    p_rot = rotate_point_cloud(positives)

    neg_files = []
    neg_indices = []
    if(len(hard_neg) == 0):
        random.shuffle(dict_value["negatives"])
        for i in range(num_neg):
            neg_files.append(QUERY_DICT[dict_value["negatives"][i]]["query"])
            neg_indices.append(dict_value["negatives"][i])
    else:
        random.shuffle(dict_value["negatives"])
        for i in hard_neg:
            neg_files.append(QUERY_DICT[i]["query"])
            neg_indices.append(i)
        j = 0
        while(len(neg_files) < num_neg):
            if not dict_value["negatives"][j] in hard_neg:
                neg_files.append(
                    QUERY_DICT[dict_value["negatives"][j]]["query"])
                neg_indices.append(dict_value["negatives"][j])
            j += 1
    negatives = load_pc_files(neg_files, True)
    n_rot = rotate_point_cloud(negatives)

    if other_neg is False:
        return [q_rot, p_rot, n_rot]

    # For Quadruplet Loss
    else:
        # get neighbors of negatives and query
        neighbors = []
        for pos in dict_value["positives"]:
            neighbors.append(pos)
        for neg in neg_indices:
            for pos in QUERY_DICT[neg]["positives"]:
                neighbors.append(pos)
        possible_negs = list(set(QUERY_DICT.keys())-set(neighbors))
        random.shuffle(possible_negs)

        if(len(possible_negs) == 0):
            return [q_jit, p_jit, n_jit, np.array([])]

        neg2 = load_pc_file(QUERY_DICT[possible_negs[0]]["query"], True)
        n2_rot = rotate_point_cloud(np.expand_dims(neg2, axis=0))
        n2_rot = np.squeeze(n2_rot)

        return [q_rot, p_rot, n_rot, n2_rot]


def get_jittered_tuple(dict_value, num_pos, num_neg, QUERY_DICT, hard_neg=[], other_neg=False):
    query = load_pc_file(dict_value["query"],True)  # Nx3
    #q_rot= rotate_point_cloud(np.expand_dims(query, axis=0))
    q_jit = jitter_point_cloud(np.expand_dims(query, axis=0))
    q_jit = np.squeeze(q_jit)

    random.shuffle(dict_value["positives"])
    pos_files = []
    for i in range(num_pos):
        pos_files.append(QUERY_DICT[dict_value["positives"][i]]["query"])
    #positives= load_pc_files(dict_value["positives"][0:num_pos])
    positives = load_pc_files(pos_files,True)
    p_jit = jitter_point_cloud(positives)

    neg_files = []
    neg_indices = []
    if(len(hard_neg) == 0):
        random.shuffle(dict_value["negatives"])
        for i in range(num_neg):
            neg_files.append(QUERY_DICT[dict_value["negatives"][i]]["query"])
            neg_indices.append(dict_value["negatives"][i])
    else:
        random.shuffle(dict_value["negatives"])
        for i in hard_neg:
            neg_files.append(QUERY_DICT[i]["query"])
            neg_indices.append(i)
        j = 0
        while(len(neg_files) < num_neg):
            if not dict_value["negatives"][j] in hard_neg:
                neg_files.append(
                    QUERY_DICT[dict_value["negatives"][j]]["query"])
                neg_indices.append(dict_value["negatives"][j])
            j += 1
    negatives = load_pc_files(neg_files,True)
    n_jit = jitter_point_cloud(negatives)

    if other_neg is False:
        return [q_jit, p_jit, n_jit]

    # For Quadruplet Loss
    else:
        # get neighbors of negatives and query
        neighbors = []
        for pos in dict_value["positives"]:
            neighbors.append(pos)
        for neg in neg_indices:
            for pos in QUERY_DICT[neg]["positives"]:
                neighbors.append(pos)
        possible_negs = list(set(QUERY_DICT.keys())-set(neighbors))
        random.shuffle(possible_negs)

        if(len(possible_negs) == 0):
            return [q_jit, p_jit, n_jit, np.array([])]

        neg2 = load_pc_file(QUERY_DICT[possible_negs[0]]["query"],True)
        n2_jit = jitter_point_cloud(np.expand_dims(neg2, axis=0))
        n2_jit = np.squeeze(n2_jit)

        return [q_jit, p_jit, n_jit, n2_jit]
