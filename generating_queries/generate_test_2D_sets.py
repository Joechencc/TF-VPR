import os
import pickle
import random
import set_path

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

import config as cfg

import scipy.io as sio
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

base_path = "/data2/cc_data"
runs_folder = "2D_data"

filename = "gt_pose.mat"

evaluate_all = False

all_folders = sorted(os.listdir(os.path.join(base_path,runs_folder)))

folders = []

# All runs are used for training (both full and partial)
if evaluate_all:
    index_list = list(range(6))
else:
    index_list = [2,4]
print("Number of runs: "+str(len(index_list)))
for index in index_list:
    folders.append(all_folders[index])
print(folders)

##########################################

def output_to_file(output, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Done ", filename)

#########################################
def construct_query_dict(df_centroids, df_database, folder_num, indice_train, indice_test, filename_train, filename_test, test=False):
    database_trees = []
    test_trees = []
    tree = KDTree(df_centroids[['x','y']])
    ind_nn = tree.query_radius(df_centroids[['x','y']],r=15)
    ind_r = tree.query_radius(df_centroids[['x','y']], r=50)
    queries_sets = []
    database_sets = []
    count_index = 0
    for folder in range(folder_num):
        queries = {}
        for i in range(indice_test[folder]):
            temp_indx = count_index + i
            query = df_centroids.iloc[temp_indx]["file"]
            #print("folder:"+str(folder))
            #print("query:"+str(query))
            queries[len(queries.keys())] = {"query":query,
                "x":float(df_centroids.iloc[temp_indx]['x']),"y":float(df_centroids.iloc[temp_indx]['y'])}
        queries_sets.append(queries)
        test_tree = KDTree(df_centroids[['x','y']])
        test_trees.append(test_tree)
        count_index = count_index + indice_test[folder]

    count_index = 0
    for folder in range(folder_num):
        dataset = {}
        for i in range(indice_train[folder]):
            temp_indx = count_index + i
            data = df_database.iloc[temp_indx]["file"]
            dataset[len(dataset.keys())] = {"query":data,
                     "x":float(df_database.iloc[temp_indx]['x']),"y":float(df_database.iloc[temp_indx]['y'])}
        database_sets.append(dataset)
        database_tree = KDTree(df_database[['x','y']])
        database_trees.append(database_tree)
        count_index = count_index + indice_train[folder]

    if test:
        for i in range(len(database_sets)):
            tree = database_trees[i]
            for j in range(len(queries_sets)):
                if(i == j):
                    continue
                for key in range(len(queries_sets[j].keys())):
                    coor = np.array(
                        [[queries_sets[j][key]["x"],queries_sets[j][key]["y"]]])
                    index = tree.query_radius(coor, r=25)
                    #print("index:"+str(index))
                    # indices of the positive matches in database i of each query (key) in test set j
                    queries_sets[j][key][i] = index[0].tolist()
    
    output_to_file(queries_sets, filename_test)
    output_to_file(database_sets, filename_train)

# Initialize pandas DataFrame
df_train = pd.DataFrame(columns=['file','x','y'])
df_test = pd.DataFrame(columns=['file','x','y'])

df_files_test = []
df_files_train =[]

indice_train = []
indice_test = []

df_locations_tr_x = []
df_locations_tr_y = []
df_locations_ts_x = []
df_locations_ts_y = []

for folder in folders:
    df_locations = sio.loadmat(os.path.join(
        base_path,runs_folder,folder,filename))
    
    df_locations = df_locations['pose']
    df_locations = torch.tensor(df_locations, dtype = torch.float).cpu()

    #2048 Database 10 testing
    test_index = random.choices(range(len(df_locations)), k=10)
    train_index = list(range(df_locations.shape[0]))
    
    indice_train.append(df_locations.shape[0])
    indice_test.append(10)
    #for i in test_index:
    #    train_index.pop(i)
    
    df_locations_tr_x.extend(list(df_locations[train_index,0]))
    df_locations_tr_y.extend(list(df_locations[train_index,1]))
    df_locations_ts_x.extend(list(df_locations[test_index,0]))
    df_locations_ts_y.extend(list(df_locations[test_index,1]))

    all_files = list(sorted(os.listdir(os.path.join(base_path,runs_folder,folder))))
    all_files.remove('gt_pose.mat')

    for (indx, file_) in enumerate(all_files):
        if indx in test_index:
            df_files_test.append(os.path.join(base_path,runs_folder,folder,file_))
        df_files_train.append(os.path.join(base_path,runs_folder,folder,file_))

df_train = pd.DataFrame(list(zip(df_files_train, df_locations_tr_x, df_locations_tr_y)),
                                               columns =['file','x', 'y'])
df_test = pd.DataFrame(list(zip(df_files_test, df_locations_ts_x, df_locations_ts_y)),
                                               columns =['file','x', 'y'])

print("Number of training submaps: "+str(len(df_train['file'])))
print("Number of non-disjoint test submaps: "+str(len(df_test['file'])))


#construct_query_dict(df_train,len(folders),"evaluation_database.pickle",False)
if not evaluate_all:
    construct_query_dict(df_test, df_train, len(folders), indice_train, indice_test, "evaluation_database.pickle", "evaluation_query.pickle", True)
else:
    construct_query_dict(df_test, df_train, len(folders), indice_train, indice_test, "evaluation_database_full.pickle", "evaluation_query_full.pickle", True)
