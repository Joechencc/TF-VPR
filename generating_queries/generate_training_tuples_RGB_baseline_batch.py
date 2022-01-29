####################################
# For training and test data split #
####################################

import os
import sys
import pickle
import random
import numpy as np
import pandas as pd
import json



def construct_dict(folder_num, df_files, df_files_all, df_indices, filename,
                   pre_dir, k_nearest, k_furthest, traj_len,
                   definite_positives=None):
    """
    Utility function to construct a dictionary
    and save it to a pickle file.
    """
    pos_index_range = list(range(-k_nearest//2, (k_nearest//2)+1))
    mid_index_range = list(range(-k_nearest, (k_nearest)+1))
    queries = {}
    count = 0
    traj_len = int(len(df_files_all)/folder_num)
    for df_indice in df_indices:
        cur_fold_num = int(df_indice//traj_len)
        file_index = int(df_indice%traj_len)
        positive_l = []
        negative_l = list(range(cur_fold_num*traj_len, (cur_fold_num+1)*traj_len, 1))

        cur_indice = df_indice % traj_len

        for index_pos in pos_index_range:
            if (index_pos + cur_indice >= 0) and (index_pos + cur_indice <= traj_len -1):
                positive_l.append(index_pos + df_indice)
        for index_pos in mid_index_range:
            if (index_pos + cur_indice >= 0) and (index_pos + cur_indice <= traj_len -1):
                negative_l.remove(index_pos + df_indice)
        #positive_l.append(df_indice)
        #positive_l.append(df_indice)
        #negative_l.remove(df_indice)

        if definite_positives is not None:
            if len(definite_positives)==1:
                if definite_positives[0][df_indice].ndim ==2:
                    positive_l.extend(definite_positives[0][df_indice][0])
                    negative_l = [i for i in negative_l if i not in definite_positives[0][df_indice][0]]
                else:
                    positive_l.extend(definite_positives[0][df_indice])
                    negative_l = [i for i in negative_l if i not in definite_positives[0][df_indice]]
            else:
                positive_l.extend(definite_positives[df_indice])
                positive_l = list(set(positive_l))
                negative_l = [i for i in negative_l if i not in definite_positives[df_indice]]

        queries[count] = {"query":df_files[count],
                        "positives":positive_l,"negatives":negative_l}
        count = count + 1
    with open(filename, 'wb') as handle:
        # lot of cpu power used here. large dict is being saved. Find a way to use GPU to save this!
        # json,
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filename)

def generate(scene_index, data_index, definite_positives=None, inside=True):
    base_path = "data/habitat_4/train/Goffs/"
    pre_dir = base_path

    # Initialize pandas DataFrame
    k_nearest = 10
    k_furthest = 50


    df_files_test = []
    df_files_train =[]
    df_files = []

    df_indices_train = []
    df_indices_test = []
    df_indices = []

    # gives you the list of sub folders to which has different images
    print(os.getcwd())
    fold_list = list(sorted(os.listdir(base_path)))
    all_files = []
    for fold in fold_list:
        # Why length of whole files including the mat file
        files_ = []
        # count .png files
        num_files = len(list(os.listdir(os.path.join(base_path, fold)))) - 1
        for ind in range(num_files):
            file_ = "panoimg_"+str(ind)+".png"
            files_.append(os.path.join(base_path, fold, file_))
        all_files.extend(files_)

    traj_len = len(all_files)

    #n Training 10 testing
    test_sample = len(fold_list)*10
    file_index = list(range(traj_len))
    test_index = random.sample(range(traj_len), k=test_sample)
    train_index = list(range(traj_len))
    for ts_ind in test_index:
        train_index.remove(ts_ind)

    for indx in range(traj_len):
        if indx in test_index:
            df_files_test.append(all_files[indx])
            df_indices_test.append(indx)
        else:
            df_files_train.append(all_files[indx])
            df_indices_train.append(indx)
        df_files.append(all_files[indx])
        df_indices.append(indx)

    queries_file_name_list = \
            ["train_pickle/training_queries_baseline_",
             "train_pickle/test_queries_baseline_",
             "train_pickle/db_queries_baseline_"]

    if inside == True:

        construct_dict(len(fold_list), df_files_train, df_files,
                       df_indices_train, queries_file_name_list[0] + \
                       str(data_index) + ".pickle", pre_dir, k_nearest,
                       k_furthest, int(traj_len/len(fold_list)))

        construct_dict(len(fold_list), df_files_test, df_files,
                       df_indices_test, queries_file_name_list[1] + \
                       str(data_index) + ".pickle", pre_dir, k_nearest,
                       k_furthest, int(traj_len/len(fold_list)))

        construct_dict(len(fold_list), df_files, df_files, df_indices,
                       queries_file_name_list[2] + str(data_index) +  \
                       ".pickle", pre_dir, k_nearest, k_furthest,
                       int(traj_len/len(fold_list)))

    else:

        construct_dict(len(fold_list), df_files_train,df_files,
                       df_indices_train, "generating_queries/" + \
                       queries_file_name_list[0] + str(data_index) + \
                       ".pickle", pre_dir, k_nearest, k_furthest,
                       int(traj_len/len(fold_list)),
                       definite_positives=definite_positives)

        construct_dict(len(fold_list), df_files_test,df_files, df_indices_test,
                       "generating_queries/" + queries_file_name_list[1] + \
                        str(data_index) + ".pickle", pre_dir, k_nearest, k_furthest,
                        int(traj_len/len(fold_list)),
                        definite_positives=definite_positives)

        construct_dict(len(fold_list), df_files,df_files, df_indices,
                       "generating_queries/" + queries_file_name_list[2] + \
                       str(data_index) + ".pickle", pre_dir, k_nearest, k_furthest,
                       int(traj_len/len(fold_list)),
                       definite_positives=definite_positives)

if __name__ == "__main__":
    for i in range(1):
        generate(0,i)
