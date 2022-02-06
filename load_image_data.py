import os
import pickle5 as pickle
import numpy as np
import random
import config as cfg
import open3d
import cv2
import random


# 360 image rotation, cut image in half. Data transformation.
def rotate_image(image, save_image = False):
    dim_1 = image.shape[1]
    cut_index = random.randint(0, dim_1-1)
    list_range = list(range(dim_1))
    new_list = list_range[cut_index:] + list_range[:cut_index]
    if save_image:
        print("Here: " + str(os.path.join('./results/visualization_2/',"before_rotate.png")))
        cv2.imwrite(os.path.join('./results/visualization_2/',"before_rotate.png"), image)
    image =  image[:, new_list, :]
    if save_image:
        cv2.imwrite(os.path.join('./results/visualization_2/',"after_rotate.png"), image)

    return image

def get_queries_dict(filename):
    print("Loading queries...")
    if filename == "" or not os.path.exists(filename):
        raise Exception("File not found or path is not correct!")
    with open(filename, 'rb') as handle:
        queries = pickle.load(handle)
        print("Filename: " + str(filename) + " Loaded.")
        return queries


def get_sets_dict(filename):
    print("Trajectories Loaded.")
    if filename == "" or not os.path.exists(filename):
        raise Exception("File not found or path is not correct!")
    with open(filename, 'rb') as handle:
        trajectories = pickle.load(handle)
        print("Filename: " + str(filename) + " Loaded.")
        return trajectories


# File path, using config file, resize dimensions
def load_image_file(filename, full_path=False):
    if full_path:
        image = cv2.imread(filename)
    else:
        print(filename)
        image = cv2.imread(os.path.join(os.getcwd(), filename))

    dim = (cfg.SIZED_GRID_X,cfg.SIZED_GRID_Y)
    image = cv2.resize(image, dim,interpolation = cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32)

    if(image.shape[2] != 3):
        print("Error in Image shape")
        return np.array([])
    return image


def load_image_files(filenames,full_path):
    images = []
    for filename in filenames:
        image = load_image_file(filename, full_path=full_path)
        images.append(image)
    images = np.asarray(images, dtype=np.float32)
    return images


def rotate_point_cloud(batch_data):
    """
    Randomly rotate the point clouds to augument the dataset
    rotation is per shape based along up direction
    Input:
        BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        #-90 to 90
        rotation_angle = (np.random.uniform()*np.pi) - np.pi/2.0
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, -sinval, 0],
                                    [sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(
            shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


# Load your query image (pos and neg).
def get_query_tuple(dict_value, num_pos, num_neg, QUERY_DICT, hard_neg=[], other_neg=False):
    """
    get query tuple for dictionary entry
    return list [query,positives,negatives]
    """
    #print("query:"+str(dict_value["query"]))
    query = load_image_file(dict_value["query"], full_path=False)  # Nx3
    random.shuffle(dict_value["positives"])
    pos_files = []

    #print("dict_value[positives]:"+str(dict_value["positives"]))
    for i in range(num_pos):
        pos_files.append(QUERY_DICT[dict_value["positives"][i]]["query"])

    #print("pos_files:"+str(pos_files))
    positives = load_image_files(pos_files,full_path=False)
    neg_files = []
    neg_indices = []
    if(len(hard_neg) == 0):
        random.shuffle(dict_value["negatives"])
        #print("dict_value[negatives]:"+str(dict_value["negatives"]))
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

    negatives = load_image_files(neg_files,full_path=False)
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

        neg2 = load_image_file(QUERY_DICT[possible_negs[0]]["query"],full_path=False)
        return [query, positives, negatives, neg2]
