import argparse
import scipy.sparse as sp
import numpy as np
import torch

def load_data(project, version,att):#modified from code: pygcn
    print('Loading {} dataset...'.format(project))

    idx_features_labels1 = np.genfromtxt("data/csv/y_{}_{}_{}.csv".format(project,version,att),delimiter = ',',
                                        dtype=np.dtype(str))
    idx_features_labels = np.genfromtxt("data/csv/feature_{}_{}_{}.csv".format(project,version,att),delimiter = ',',
                                        dtype = np.dtype(str))
    idx_features_labels2 = np.genfromtxt("data/csv/feature_{}_{}_{}.csv".format(project,version,att), delimiter = ',',
                                        dtype = np.int32)
    features = sp.csr_matrix(idx_features_labels[:, 1:], dtype=np.float64)
    # print(features)
    labels = idx_features_labels1[:]
    set_labels = set(labels)
    classes_dict = {'0':0,'1':1}

    #ipdb.set_trace()
    labels = np.array(list(map(classes_dict.get, labels)))
    # build graph

    idx = np.array(idx_features_labels2[:,0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("data/csv/edgecites_{}_{}_{}.csv".format(project,version,att),delimiter = ',',
                                    dtype=np.int32)
    edges_unordered = edges_unordered[:]

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    edges = torch.tensor(edges, dtype = torch.int64).T
    return adj, features, labels,edges


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
