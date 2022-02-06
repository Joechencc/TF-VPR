import argparse
import math
import numpy as np
import socket
import importlib
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
import torch
import torch.nn as nn
from torch.backends import cudnn

from sklearn.neighbors import KDTree

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from load_image_data import *
import models.PointNetVlad as PNV
from tensorboardX import SummaryWriter
import loss.pointnetvlad_loss

import config as cfg

cudnn.enabled = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate():
    model = PNV.PointNetVlad(global_feat=True, feature_transform=True,
                             max_pool=False, output_dim=cfg.FEATURE_OUTPUT_DIM,
                             num_points=cfg.NUM_POINTS)
    model = model.to(device)

    resume_filename = cfg.LOG_DIR + "checkpoint.pth.tar"
    print("Resuming From ", resume_filename)
    checkpoint = torch.load(resume_filename)
    saved_state_dict = checkpoint['state_dict']
    model.load_state_dict(saved_state_dict)

    #model = nn.DataParallel(model)
    ave_one_percent_recall = evaluate_model(model)
    print("ave_one_percent_recall:"+str(ave_one_percent_recall))


def evaluate_model(model,optimizer,epoch,scene_index,save=False,full_pickle=False):
    if save:
        torch.save({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            }, cfg.LOG_DIR + "checkpoint.pth.tar")

    print("epoch:"+str(epoch))
    if full_pickle:
        DATABASE_SETS = get_sets_dict('generating_queries/evaluation_database_full.pickle')
        QUERY_SETS = get_sets_dict('generating_queries/evaluation_query_full.pickle')
    else:
        DATABASE_SETS = get_sets_dict(cfg.EVAL_DATABASE_FILE)
        QUERY_SETS = get_sets_dict(cfg.EVAL_QUERY_FILE)

    if not os.path.exists(cfg.RESULTS_FOLDER):
        os.mkdir(cfg.RESULTS_FOLDER)

    DATABASE_VECTORS = []
    QUERY_VECTORS = []

    for i in range(len(DATABASE_SETS)):
        DATABASE_VECTORS.append(get_latent_vectors(model, DATABASE_SETS[i]))

    for j in range(len(QUERY_SETS)):
        QUERY_VECTORS.append(get_latent_vectors(model, QUERY_SETS[j]))

    recall_1 = 0
    recall_5 = 0
    recall_10 = 0

    # Save Evaluate vectors
    if full_pickle:
        return DATABASE_VECTORS
    else:
        file_name = os.path.join(cfg.RESULTS_FOLDER, "database"+str(epoch)+".npy")
        np.save(file_name, np.array(DATABASE_VECTORS))
        print("saving for DATABASE_VECTORS to "+str(file_name))

    pair_recall_1, pair_recall_5, pair_recall_10= get_recall(
        0, 0, DATABASE_VECTORS, QUERY_VECTORS, QUERY_SETS)
    recall_1 = np.array(pair_recall_1)
    recall_5 = np.array(pair_recall_5)
    recall_10 = np.array(pair_recall_10)

    print("recall_1:"+str(recall_1))
    ave_recall_1 = recall_1 #/ count
    ave_recall_5 = recall_5 #/ count
    ave_recall_10 = recall_10 #/ count
    with open(os.path.join(cfg.OUTPUT_FILE), "w") as output:
        output.write("Average Recall @1:\n")
        output.write(str(ave_recall_1)+"\n")
        output.write("Average Recall @5:\n")
        output.write(str(ave_recall_5)+"\n")
        output.write("Average Recall @10:\n")
        output.write(str(ave_recall_10)+"\n")
        output.write("\n\n")
    return ave_recall_1, ave_recall_5, ave_recall_10



def get_latent_vectors(model, dict_to_process):
    model.eval()
    is_training = False
    train_file_idxs = np.arange(0, len(dict_to_process.keys()))

    batch_num = cfg.EVAL_BATCH_SIZE * \
        (1 + cfg.EVAL_POSITIVES_PER_QUERY + cfg.EVAL_NEGATIVES_PER_QUERY)
    q_output = []
    for q_index in range(len(train_file_idxs)//batch_num):
        file_indices = train_file_idxs[q_index *
                                       batch_num:(q_index+1)*(batch_num)]
        file_names = []
        for index in file_indices:
            file_names.append(dict_to_process[index]["query"])

        queries = load_image_files(file_names,False)

        with torch.no_grad():
            feed_tensor = torch.from_numpy(queries).float()
            feed_tensor = feed_tensor.unsqueeze(1)
            feed_tensor = feed_tensor.to(device)
            out= model(feed_tensor)

        out = out.detach().cpu().numpy()
        out = np.squeeze(out)

        q_output.append(out)

    q_output = np.array(q_output)
    if(len(q_output) != 0):
        q_output = q_output.reshape(-1, q_output.shape[-1])

    # handle edge case
    index_edge = len(train_file_idxs) // batch_num * batch_num
    if index_edge < len(dict_to_process.keys()):
        file_indices = train_file_idxs[index_edge:len(dict_to_process.keys())]
        file_names = []

        for index in file_indices:
            file_names.append(dict_to_process[index]["query"])
        queries = load_image_files(file_names,False)

        with torch.no_grad():
            feed_tensor = torch.from_numpy(queries).float()
            feed_tensor = feed_tensor.unsqueeze(1)
            feed_tensor = feed_tensor.to(device)
            o1 = model(feed_tensor)

        output = o1.detach().cpu().numpy()
        output = np.squeeze(output)
        if (q_output.shape[0] != 0):
            q_output = np.vstack((q_output, output))
        else:
            q_output = output

    model.train()

    return q_output


def get_recall(m, n, DATABASE_VECTORS, QUERY_VECTORS, QUERY_SETS):
    database_output = DATABASE_VECTORS[m]  #2048*256
    queries_output = QUERY_VECTORS[n]      #10*256

    database_nbrs = KDTree(database_output)

    recalls = []
    similarity_scores = []
    N_percent_recalls = []

    n_values = [1,5,10,20]
    for value in n_values:
        num_evaluated = 0
        recall_N_per = 0
        for i in range(len(queries_output)):
            true_neighbors = QUERY_SETS[n][i][m]
            if(len(true_neighbors) == 0):
                continue
            num_evaluated += 1
            distances, indices = database_nbrs.query(
                np.array([queries_output[i]]),k=value+11)
            compare_a = set(indices[0][0:50].tolist()[:value])
            k_nearest = 10
            pos_index_range = list(range(-k_nearest//2, (k_nearest//2)+1))
            for pos_index in pos_index_range:
                try:
                    compare_a.remove(pos_index+i)
                except:
                    pass
            compare_b = set(true_neighbors)
            try:
                compare_b.remove(i)
            except:
                pass
            if len(list(compare_a.intersection(compare_b))) > 0:
                recall_N_per += 1

        if float(num_evaluated)!=0:
            recall_N = (recall_N_per/float(num_evaluated))*100
        else:
            recall_N = 0
        recalls.append(recall_N)
    recall_1, recall_5, recall_10 = recalls[0], recalls[1], recalls[2]

    return recall_1, recall_5, recall_10

if __name__ == "__main__":
    if not os.path.isdir(cfg.RESULTS_FOLDER):
        os.mkdir(cfg.RESULTS_FOLDER)

    evaluate()
