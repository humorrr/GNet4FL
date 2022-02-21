import argparse
import scipy.sparse as sp
import numpy as np
import torch
import ipdb
from scipy.io import loadmat
import networkx as nx
import multiprocessing as mp
import torch.nn.functional as F
from functools import partial
import random
from sklearn.metrics import roc_auc_score, f1_score
from copy import deepcopy
from scipy.spatial.distance import pdist, squareform


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action = 'store_true', default = False,
                        help = 'Disables CUDA training.')
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--nhid', type = int, default = 100)
    parser.add_argument('--epochs', type = int, default = 1,
                        help = 'Number of epochs to train.')
    parser.add_argument('--lr', type = float, default = 0.04)
    parser.add_argument('--weight_decay', type = float, default = 5e-5)
    parser.add_argument('--dropout', type = float, default = 0.02)
    return parser


def split_genuine(labels):
    # labels: n-dim Longtensor, each element in [0,...,m-1].
    # cora: m=7
    num_classes = len(set(labels.tolist()))
    c_idxs = []  # class-wise index
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes, 3)).astype(int)

    for i in range(num_classes):
        c_idx = (labels == i).nonzero()[:, -1].tolist()
        print("c_idx:",c_idx)
        c_num = len(c_idx)
        print('{:d}-th class sample number: {:d}'.format(i, len(c_idx)))
        random.shuffle(c_idx)
        c_idxs.append(c_idx)

        if c_num < 4:
            if c_num < 3:
                print("too small class type")
                ipdb.set_trace()
            c_num_mat[i, 0] = 1
            c_num_mat[i, 1] = 1
            c_num_mat[i, 2] = 1
        else:
            print("elsec_num")
            c_num_mat[i, 0] = int(c_num / 4)
            c_num_mat[i, 1] = int(c_num / 3)
            c_num_mat[i, 2] = int(c_num / 2)
        print("c_num_mat:",c_num_mat)
        train_idx = train_idx + c_idx[:c_num_mat[i, 0]]

        val_idx = val_idx + c_idx[c_num_mat[i, 0]:c_num_mat[i, 0] + c_num_mat[i, 1]]
        test_idx = test_idx + c_idx[
                              c_num_mat[i, 0] + c_num_mat[i, 1]:c_num_mat[i, 0] + c_num_mat[i, 1] + c_num_mat[i, 2]]
    # print("train_idx:", train_idx, val_idx, test_idx)
    # print("train_idx:", len(train_idx), len(val_idx), len(test_idx))
    random.shuffle(train_idx)

    # ipdb.set_trace()

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)
    # c_num_mat = torch.LongTensor(c_num_mat)
    # print(c_num_mat)
    return train_idx, val_idx, test_idx, c_num_mat


def print_edges_num(dense_adj, labels):
    c_num = labels.max().item() + 1
    dense_adj = np.array(dense_adj)
    labels = np.array(labels)

    for i in range(c_num):
        for j in range(c_num):
            # ipdb.set_trace()
            row_ind = labels == i
            col_ind = labels == j

            edge_num = dense_adj[row_ind].transpose()[col_ind].sum()
            print("edges between class {:d} and class {:d}: {:f}".format(i, j, edge_num))


