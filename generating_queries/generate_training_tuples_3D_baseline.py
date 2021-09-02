import os
import sys
import pickle
import random
import set_path

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors

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

def construct_dict(df_files, filename, folder_sizes, all_folder_sizes, folder_num, all_folders, pre_dir, definite_positives=None):
    queries = {}
    for num in range(folder_num):
        #print("df_files:"+str(len(df_files)))
        if num == 0:
            overhead = 0
        else:
            overhead = 0
            for i in range(num):
                overhead = overhead + folder_sizes[i]
        #q_range = np.arange(folder_sizes[num])
        #q_range = q_range + overhead
        #print("q_range:"+str(q_range))
        df_centroids = df_files[overhead:overhead + folder_sizes[num]]
        #assert(0)
        tree = KDTree(df_centroids[['x','y']])
        ind_nn_tree = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(df_centroids[['x','y']])
        ind_r_tree = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(df_centroids[['x','y']])
        distance_nn, ind_nn = ind_nn_tree.kneighbors(df_centroids[['x','y']])
        distance_r, ind_r = ind_r_tree.kneighbors(df_centroids[['x','y']])
        #ind_r = tree.query_radius(df_centroids[['x','y']], r=50)

        for i in range(len(df_centroids)):
            radius = 0.5
            #ind_nn = tree.query_radius(df_centroids[['x','y']],r=radius)
            #ind_nn = NearestNeighbors(n_neighbors=8, algorithm='ball_tree').fit(df_centroids['x','y'])
            query = df_centroids.iloc[i]["file"]
            positives = np.setdiff1d(ind_nn[i],[i]).tolist()
            negatives = np.setdiff1d(df_centroids.index.values.tolist(),ind_r[i]).tolist()
            random.shuffle(negatives)

            '''
            while(len(positives)<3):
                radius = radius+0.1
                ind_nn = tree.query_radius(df_centroids[['x','y']],r=radius)
                ind_r = tree.query_radius(df_centroids[['x','y']], r=3*radius)
                query = df_centroids.iloc[i]["file"]
                positives = np.setdiff1d(ind_nn[i],[i]).tolist()
                negatives = np.setdiff1d(
                            df_centroids.index.values.tolist(),ind_r[i]).tolist()
                random.shuffle(negatives)
            count = 0
            while(len(positives)>8):
                radius = radius-0.0005
                if radius <= 0:
                    print("radius:"+str(radius))
                    while(radius <= 0):
                        radius = radius + 0.005
                ind_nn = tree.query_radius(df_centroids[['x','y']],r=radius)
                ind_r = tree.query_radius(df_centroids[['x','y']], r=3*radius)
                query = df_centroids.iloc[i]["file"]
                positives = np.setdiff1d(ind_nn[i],[i]).tolist()
                negatives = np.setdiff1d(df_centroids.index.values.tolist(),ind_r[i]).tolist()
                random.shuffle(negatives)
                count= count +1
                if count>=1000:
                    assert(0)
            
            if (len(positives)>=3 and len(positives)<=8):
                pass
            else:
                print(len(positives))
                if len(positives)<3:
                    while(len(positives)<3):
                        radius = radius+0.0001
                        ind_nn = tree.query_radius(df_centroids[['x','y']],r=radius)
                        ind_r = tree.query_radius(df_centroids[['x','y']], r=3*radius)
                        query = df_centroids.iloc[i]["file"]
                        positives = np.setdiff1d(ind_nn[i],[i]).tolist()
                        negatives = np.setdiff1d(
                                df_centroids.index.values.tolist(),ind_r[i]).tolist()
                        random.shuffle(negatives)

                elif len(positives)>8:
                    count = 0
                    while(len(positives)>8):
                        radius = radius-0.0001
                        if radius <= 0:
                            print("radius:"+str(radius))
                            while(radius <= 0):
                                radius = radius + 0.005
                        ind_nn = tree.query_radius(df_centroids[['x','y']],r=radius)
                        ind_r = tree.query_radius(df_centroids[['x','y']], r=3*radius)
                        query = df_centroids.iloc[i]["file"]
                        positives = np.setdiff1d(ind_nn[i],[i]).tolist()
                        negatives = np.setdiff1d(df_centroids.index.values.tolist(),ind_r[i]).tolist()
                        random.shuffle(negatives)
                        count= count +1
                        if count>=1000:
                            assert(0)
            assert(len(positives)>=3 and len(positives)<=8)
            '''
            queries[i+overhead] = {"query":df_centroids.iloc[i]['file'],
                          "positives":positives,"negatives":negatives}
            #print("query:"+str(query))
            #print("df_centroids.index.values.tolist():"+str(len(df_centroids.index.values.tolist())))
            #print("positives:"+str(len(positives)))
            #print("negatives:"+str(len(negatives)))
            '''
            max_dis = 0
            for pos in positives:
                x_delta = df_centroids.iloc[pos+overhead]['x'] - df_centroids.iloc[i]['x']
                y_delta = df_centroids.iloc[pos+overhead]['y'] - df_centroids.iloc[i]['y']
                dis = math.sqrt(x_delta**2 + y_delta**2)
                print("dis:"+str(dis))
            if dis > max_dis:
                max_dis = dis
            '''
            #print("negatives:"+str(len(negatives)))
    with open(filename, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", filename)

def generate(data_index, definite_positives=None, inside=True):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    base_path = "/data2"
    runs_folder = "cc_data/"
    
    folders_ = os.path.join(base_path, runs_folder)
    '''
    all_folder_360 = sorted(os.listdir(os.path.join(folders_, "360")))
    all_folder_361 = sorted(os.listdir(os.path.join(folders_, "361")))
    
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
    df_locations_x = []
    df_locations_y = []

    pre_dir = os.path.join(folders_, "2D_data")
    folders = sorted(os.listdir(os.path.join(folders_, "2D_data")))
    all_folders = folders
    folder_num = len(folders)

    print("folder_num:"+str(folder_num))
    print("folders:"+str(folders))
    folder_sizes_train = []
    folder_sizes_test = []
    folder_sizes = []
    filename = "gt_pose.mat"

    for folder in folders:
        df_locations = sio.loadmat(os.path.join(
                       pre_dir,folder,filename))
        
        df_locations = df_locations['pose']
        df_locations = torch.tensor(df_locations, dtype = torch.float).cpu()
        
        #n Training 10 testing
        file_index = list(range(df_locations.shape[0]))
        test_index = random.sample(range(len(df_locations)), k=10)
        train_index = list(range(df_locations.shape[0]))
        for ts_ind in test_index:
            train_index.remove(ts_ind)
        
        folder_sizes_train.append(len(train_index))
        folder_sizes_test.append(10)
        folder_sizes.append(df_locations.shape[0])

        df_locations_tr_x.extend(list(df_locations[train_index,0]))
        df_locations_tr_y.extend(list(df_locations[train_index,1]))
        df_locations_ts_x.extend(list(df_locations[test_index,0]))
        df_locations_ts_y.extend(list(df_locations[test_index,1]))
        df_locations_x.extend(list(df_locations[file_index,0]))
        df_locations_y.extend(list(df_locations[file_index,1]))

        all_files = list(sorted(os.listdir(os.path.join(pre_dir,folder))))
        all_files.remove('gt_pose.mat')

        for (indx, file_) in enumerate(all_files): 
            if indx in test_index:
                df_files_test.append(os.path.join(pre_dir,folder,file_))
            else:
                df_files_train.append(os.path.join(pre_dir,folder,file_))
            df_files.append(os.path.join(pre_dir,folder,file_))

    #print("df_locations_tr_x:"+str(len(df_locations_tr_x)))
    #print("df_files_test:"+str(len(df_files_test)))
    df_train = pd.DataFrame(list(zip(df_files_train, df_locations_tr_x, df_locations_tr_y)),
                                                           columns =['file','x', 'y'])
    df_test = pd.DataFrame(list(zip(df_files_test, df_locations_ts_x, df_locations_ts_y)),
                                                           columns =['file','x', 'y'])
    df_files = pd.DataFrame(list(zip(df_files, df_locations_x, df_locations_y)),
                                                           columns =['file','x', 'y'])
    
    #print("Number of training submaps: "+str(len(df_train['file'])))
    #print("Number of non-disjoint test submaps: "+str(len(df_test['file'])))

    #print("df_train:"+str(len(df_train)))
    #print("df_test:"+str(len(df_test)))
    #pre_dir = os.path.join(pre_dir,runs_folder)

    if inside == True:
        construct_dict(df_train,"training_queries_baseline.pickle", folder_sizes_train, folder_sizes, folder_num, all_folders, pre_dir)
        construct_dict(df_test, "test_queries_baseline.pickle", folder_sizes_test, folder_sizes, folder_num, all_folders, pre_dir)
        construct_dict(df_files, "db_queries_baseline.pickle", folder_sizes, folder_sizes, folder_num, all_folders, pre_dir)
    else:
        construct_dict(df_train, "generating_queries/training_queries_baseline.pickle", folder_sizes_train, folder_sizes, folder_num, all_folders, pre_dir, definite_positives=definite_positives)
        construct_dict(df_test, "generating_queries/test_queries_baseline.pickle", folder_sizes_test, folder_sizes, folder_num, all_folders, pre_dir, definite_positives=definite_positives)
        construct_dict(df_files, "generating_queries/db_queries_baseline.pickle", folder_sizes, folder_sizes, folder_num, all_folders, pre_dir, definite_positives=definite_positives)

if __name__ == "__main__":
    for i in range(1):
        generate(i)
