import os
import sys
import pickle
import random
import set_path

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

import config as cfg

import scipy.io as sio
import torch


from loading_pointclouds import *
sys.path.append('/usr/local/lib/python3.6/dist-packages/python_pcl-0.3-py3.6-linux-x86_64.egg/')
import pcl as pcl_lib

#####For training and test data split#####

def check_in_test_set(northing, easting, points, x_width, y_width):
    in_test_set = False
    for point in points:
        if(point[0]-x_width < northing and northing < point[0]+x_width and point[1]-y_width < easting and easting < point[1]+y_width):
            in_test_set = True
            break
    return in_test_set
##########################################


def construct_query_dict(df_files, df_indice, filename):
    queries = {}
    
    for i in range(len(ind_nn)):
        query = df_centroids.iloc[i]["file"]
        positives = np.setdiff1d(ind_nn[i],[i]).tolist()
        negatives = np.setdiff1d(
            df_centroids.index.values.tolist(),ind_r[i]).tolist()
        random.shuffle(negatives)
        queries[i] = {"query":df_centroids.iloc[i]['file'],
                      "positives":positives,"negatives":negatives}

    with open(filename, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", filename)

def construct_dict(df_files, df_indices, filename, folder_sizes, all_folder_sizes, folder_num, all_folders, pre_dir, k_nearest, k_furthest, definite_positives=None):
    pos_index_range = list(range(-k_nearest//2, (k_nearest//2)+1))
    mid_index_range = list(range(-k_nearest, (k_nearest)+1))
    
    count_index = 0
    
    queries = {}
    for num in range(folder_num):
        folder = os.path.join(pre_dir,all_folders[num])
        '''
        gt_mat = os.path.join(folder, 'gt_pose.mat')
        df_locations = sio.loadmat(gt_mat)
        df_locations = df_locations['pose']
        df_locations = torch.tensor(df_locations, dtype = torch.float).cpu()
        '''
        for index in range(folder_sizes[num]):
            #print("num:"+str(num))
            #print("count_index:"+str(count_index))
            df_indice = df_indices[count_index + index]
            positive_l = []
            negative_l = list(range(all_folder_sizes[num]))
            
            for index_pos in pos_index_range:
                if (index_pos + df_indice >= 0) and (index_pos + df_indice <= all_folder_sizes[num] -1):
                    positive_l.append(index_pos + df_indice)
            for index_pos in mid_index_range:
                if (index_pos + df_indice >= 0) and (index_pos + df_indice <= all_folder_sizes[num] -1):
                    negative_l.remove(index_pos + df_indice)
            if definite_positives is not None:
                positive_l.extend(definite_positives[num][df_indice])
                negative_l = [i for i in negative_l if i not in definite_positives[num][df_indice]]

            positive_l = list(set(positive_l))
        

            '''
            negative_l_sampled = random.sample(negative_l, k=k_furthest)
            negative_l_sampled, replace_count = check_negatives(negative_l_sampled, index, df_locations, mid_index_range)
            while (replace_count!=0):
                negative_l_sampled_new = random.sample(negative_l, k=replace_count)
                negative_l_sampled_new, replace_count = check_negatives(negative_l_sampled_new, index, df_locations, mid_index_range)
                negative_l_sampled.extend(negative_l_sampled_new)
            
            replace_counts = replace_counts + replace_count
            '''
            queries[count_index + index] = {"query":df_files[count_index + index],
                          "positives":positive_l,"negatives":negative_l}
            
            '''
            print("df_files[num * (len(df_indices)//folder_num) + index]:"+str(df_files[num * (len(df_indices)//folder_num) + index]))
            
            print("positive_l:"+str(positive_l))
            print("negative_l:"+str(negative_l))
        
            assert(0)
            '''
        count_index = count_index + folder_sizes[num]
    #print("replace_counts:"+str(replace_counts))        
    #print("queries:"+str(queries[0]))
    #print("queries:"+str(queries[0][0]))

    with open(filename, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", filename)

def generate(data_index, definite_positives=None, inside=True):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    base_path = "/data2"
    runs_folder = "cc_data/"
    
    folders_ = os.path.join(base_path, runs_folder)
    
    all_folder_360 = sorted(os.listdir(os.path.join(folders_, "360")))
    all_folder_361 = sorted(os.listdir(os.path.join(folders_, "361")))
    '''
    outfile = "/home/cc/2D_data/"
      
    out_index = 0
   
    for file_360 in all_folder_360:
        pcl, gt = load_log_files(os.path.join(os.path.join(folders_, "360"), file_360), 360)
        outfolder = os.path.join(outfile,"run_"+str(out_index))
        for point in range(pcl.shape[0]):
            pc2 = pcl_lib.PointCloud(pcl[point])
            #writer = pcl_lib.io.PCDWriter()
            pcl_lib.save(pc2, os.path.join(outfolder, '{0:04}'.format(point)+".pcd"))
        
        gt_dict = {"pose":gt}
        sio.savemat(os.path.join(outfolder, "gt_pose.mat"), gt_dict)
        out_index = out_index +1

    for file_361 in all_folder_361:
        pcl, gt = load_log_files(os.path.join(os.path.join(folders_, "361"), file_361), 360)
        outfolder = os.path.join(outfile,"run_"+str(out_index))
        for point in range(pcl.shape[0]):
            pc2 = pcl_lib.PointCloud(pcl[point])
            #writer = pcl_lib.io.PCDWriter()
            pcl_lib.save(pc2, os.path.join(outfolder, '{0:04}'.format(point)+".pcd"))
        
        gt_dict = {"pose":gt}
        sio.savemat(os.path.join(outfolder, "gt_pose.mat"), gt_dict)
        out_index = out_index + 1
    assert(0)
    '''
    # Initialize pandas DataFrame

    df_train = pd.DataFrame(columns=['file','x','y'])
    df_test = pd.DataFrame(columns=['file','x','y'])

    df_files_test = []
    df_files_train =[]
    df_files = []

    df_locations_tr_x = []
    df_locations_tr_y = []
    df_locations_ts_x = []
    df_locations_ts_y = []

    pre_dir = os.path.join(folders_, "2D_data")
    folders = sorted(os.listdir(os.path.join(folders_, "2D_data")))
    all_folders = folders
    folder_num = len(folders)

    folder_sizes_train = []
    folder_sizes_test = []
    folder_sizes = []

    for folder in folders:
        df_locations = sio.loadmat(os.path.join(
                       pre_dir,folder,filename))

        all_files = list(sorted(os.listdir(os.path.join(pre_dir,folder))))
        all_files.remove('gt_pose.mat')
        
        folder_size = len(all_files)
        folder_sizes_train.append(folder_size-10)
        folder_sizes_test.append(10)
        folder_sizes.append(folder_size)

        test_index = random.sample(range(folder_size), k=10)
        train_index = list(range(folder_size))
        for ts_ind in test_index:
            train_index.remove(ts_ind)
        
        for (indx, file_) in enumerate(all_files): 
            if indx in test_index:
                df_files_test.append(os.path.join(pre_dir,folder,file_))
                df_indices_test.append(indx)
            else:
                df_files_train.append(os.path.join(pre_dir,folder,file_))
                df_indices_train.append(indx)
            df_files.append(os.path.join(pre_dir,folder,file_))
            df_indices.append(indx)
    #pre_dir = os.path.join(pre_dir,runs_folder)
    

    if inside == True:
        construct_dict(df_files_train, df_indices_train, "train_pickle/training_queries_baseline_"+str(data_index)+".pickle", folder_sizes_train, folder_sizes, folder_num, all_folders, pre_dir, k_nearest, k_furthest)
        construct_dict(df_files_test, df_indices_test, "train_pickle/test_queries_baseline_"+str(data_index)+".pickle", folder_sizes_test, folder_sizes, folder_num, all_folders, pre_dir, k_nearest, k_furthest)
        construct_dict(df_files, df_indices, "train_pickle/db_queries_baseline_"+str(data_index)+".pickle", folder_sizes, folder_sizes, folder_num, all_folders, pre_dir, k_nearest, k_furthest)
    else:
        construct_dict(df_files_train, df_indices_train, "generating_queries/train_pickle/training_queries_baseline_"+str(data_index)+".pickle", folder_sizes_train, folder_sizes, folder_num, all_folders, pre_dir, k_nearest, k_furthest, definite_positives=definite_positives)
        construct_dict(df_files_test, df_indices_test, "generating_queries/train_pickle/test_queries_baseline_"+str(data_index)+".pickle", folder_sizes_test, folder_sizes, folder_num, all_folders, pre_dir, k_nearest, k_furthest, definite_positives=definite_positives)
        construct_dict(df_files, df_indices, "generating_queries/train_pickle/db_queries_baseline_"+str(data_index)+".pickle", folder_sizes, folder_sizes, folder_num, all_folders, pre_dir, k_nearest, k_furthest, definite_positives=definite_positives)
if __name__ == "__main__":
    for i in range(20):
        generate(i)
