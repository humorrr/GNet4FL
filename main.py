import csv
import pickle
import time
import argparse
import numpy as np
import os
from sklearn.metrics import f1_score
from imblearn.under_sampling import ClusterCentroids
from sklearn.manifold import TSNE
# from Attention import attention
from torch_geometric.data import DataLoader

from MS import model
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

import models
import utils
import data_load
from sklearn import linear_model
import random
import ipdb
import copy
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE, ADASYN

# Training setting

parser = utils.get_parser()

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

project = 'Mockito'
version='1'
# list_file1 = open('data/pkl/linepo_{}_{}.pickle'.format(project,version), 'rb')
# lineset = pickle.load(list_file1)
# # print(lineset)

# list_file2 = open('data/pkl/treetoken_num_{}_{}.pickle'.format(project,version), 'rb')
# treetokennum = pickle.load(list_file2)
# # print(len(treetokennum))
# list_file3 = open('data/pkl/buggy_lines_{}_{}.pickle'.format(project,version), 'rb')
# bug_line = pickle.load(list_file3)
adj_train, features_train, labels_train ,edges_train = data_load.load_data(project,version,'train')
adj_test, features_test, labels_test ,edges_test = data_load.load_data(project,version,'test')
# print("ffeature:", labels)
# print("before append:", adj.shape, len(labels))
# print("edges:",edges,edges.shape)
#————reduction
# X = np.array(features)
# tsne = TSNE(n_components = 3)
# tsne.fit_transform(X)
# features = tsne.embedding_
# features = torch.tensor(features, dtype = torch.float32)
# print(tsne.embedding_)
# print("after shape:", adj.shape)
# print("after feature:", features.shape)
# tsne_data = open('data/pkl/t-SNE{}.pickle'.format(project), 'wb')
# pickle.dump(features, tsne_data)
# tsne_data.close()

#If the feature has been dimensionalized, perform the following
file_tsne_train = open('data/pkl/t-SNE{}_{}_{}.pickle'.format(project,version,'train'), 'rb')
features_train = pickle.load(file_tsne_train)
file_tsne_test = open('data/pkl/t-SNE{}_{}_{}.pickle'.format(project,version,'test'), 'rb')
features_test = pickle.load(file_tsne_test)
#—————failed att
def conactFeature(project,version,att,features):

    file_temp = open('data/pkl/alltimes_{}_{}_{}.pickle'.format(project,version,att), 'rb')
    times = pickle.load(file_temp)

    print(times,len(times))
    tensor_times=[]
    for i in range(0,len(times)):
        temp_times=[times[i]]
        tensor_times.append(temp_times)
    tensor_times=torch.Tensor(tensor_times)
    features=torch.cat((features,tensor_times),1)
    # features = data_load.normalize(features)
    features=torch.Tensor(features)
    return features
features_train=conactFeature(project,version,'train',features_train)
features_test=conactFeature(project,version,'test',features_test)
#————————————————————
idx_train, idx_val, class_num_mat = utils.split_genuine(labels_train)
print(idx_train, idx_val)
# Model and optimizer
encoder = models.Sage_En(nfeat = features_train.shape[1],
                         nhid = args.nhid,
                         nembed = args.nhid,
                         dropout = args.dropout)
classifier = models.Sage_Classifier(nembed = args.nhid,
                                    nhid = args.nhid,
                                    nclass = labels_train.max().item() + 1,
                                    dropout = args.dropout)
decoder = models.Decoder(nembed = args.nhid,
                         dropout = args.dropout)
optimizer_en = optim.Adam(encoder.parameters(),
                          lr = args.lr, weight_decay = args.weight_decay)
optimizer_cls = optim.Adam(classifier.parameters(),
                           lr = args.lr, weight_decay = args.weight_decay)
optimizer_de = optim.Adam(decoder.parameters(),
                          lr = args.lr, weight_decay = args.weight_decay)
print(args.cuda)
# adj=edges
if args.cuda:
    encoder = encoder.cuda()
    classifier = classifier.cuda()
    decoder = decoder.cuda()
    features_train = features_train.cuda()
    adj_train = adj_train.cuda()
    labels_train = labels_train.cuda()
    features_test = features_test.cuda()
    adj_test = adj_test.cuda()
    labels_test = labels_test.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
    print("Use", torch.cuda.device_count(), 'gpus')
    encoder = nn.DataParallel(encoder)
    classifier = nn.DataParallel(classifier)
    decoder = nn.DataParallel(decoder)
decoder.to(device)
encoder.to(device)
classifier.to(device)


def train(epoch):
    t = time.time()
    encoder.train()
    classifier.train()
    decoder.train()
    optimizer_en.zero_grad()
    optimizer_cls.zero_grad()
    optimizer_de.zero_grad()

    embed = encoder(features_train, adj_train)

    labels_new = labels_train #include train,val
    idx_train_new = idx_train
    adj_new = adj_train

    # ipdb.set_trace()
    output = classifier(embed, adj_new)
    # print("output:",output.shape)
    loss_train = F.cross_entropy(output[idx_train_new], labels_new[idx_train_new])
    # print("train len:",len(idx_train))
    # acc_train = utils.accuracy(lineset, treetokennum, output[idx_train_new],bug_line)
    if args.cuda:
        acc_train = f1_score(output[idx_train_new].max(1)[1].type_as(labels_train.cpu()),labels_new[idx_train_new].cpu(), average = 'weighted')
    else:
        acc_train = f1_score(output[idx_train_new].max(1)[1].type_as(labels_train), labels_new[idx_train_new],average='weighted')
    loss = loss_train
    loss_rec = loss_train

    loss.backward()
    optimizer_en.step()

    optimizer_cls.step()

    loss_val = F.cross_entropy(output[idx_val], labels_train[idx_val])
    # acc_val = utils.accuracy(lineset, treetokennum, output[idx_val],bug_line)
    if args.cuda:
        acc_val = f1_score(output[idx_val].max(1)[1].type_as(labels_train.cpu()), labels_train[idx_val].cpu())
    else:
        acc_val = f1_score(output[idx_val].max(1)[1].type_as(labels_train), labels_train[idx_val])

    # ipdb.set_trace()
    utils.print_class_acc(output[idx_val], labels_train[idx_val])

    print('Epoch: {:05d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'loss_rec: {:.4f}'.format(loss_rec.item()),
          # 'acc_train: {:.4f}'.format(acc_train.item()),
          'acc_train: {:.4f}'.format(acc_train),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val),
          'time: {:.4f}s'.format(time.time() - t))
    return embed


def test(epoch = 0):
    encoder.eval()
    classifier.eval()
    decoder.eval()
    embed = encoder(features_test, adj_test)
    # print("embed:", embed)
    # print("embed:", embed.shape)
    output = classifier(embed, adj_test)
    # print("test len:", len(idx_test))
    loss_test = F.cross_entropy(output[:], labels_test[:])
    # acc_test =  utils.accuracy(lineset, treetokennum, output[idx_test],bug_line)
    if args.cuda:
        acc_test =  f1_score(output[:].max(1)[1].type_as(labels_test.cpu()), labels_test[:].cpu(),average='weighted')
    else:
        acc_test =  f1_score(output[:].max(1)[1].type_as(labels_test), labels_test[:],average='weighted')
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test))

    utils.print_class_acc(output[:], labels_test[:], pre = 'test')
    return embed

t_total = time.time()
for epoch in range(args.epochs):
    embed_train = train(epoch)
    if epoch % 10 == 0:
        embed_test=test(epoch)

    # if epoch % 20 == 0:
    #     save_model(epoch)
print("feature final:", features_test)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

new_features_train = embed_train
new_features_test = embed_test

def semantic(feature):
    if args.cuda:
        mode = model.SemanticAttention(in_size = int(feature.shape[1])).to(device)
    else:
        mode = model.SemanticAttention(in_size = int(feature.shape[1]))
    mode.train()
new_features_train = semantic(new_features_train)
new_features_test = semantic(new_features_test)

if args.cuda:
    new_features_train = new_features_train.cuda().data.cpu().numpy()
    new_features_test = new_features_test.cuda().data.cpu().numpy()
else:
    new_features_train = new_features_train.detach().numpy()
    new_features_test = new_features_test.detach().numpy()
# print("new_features shape:",new_features_test.shape)
list_file1 = open('data/pkl/test/linepo_{}_{}_{}.pickle'.format(project,version,'train'), 'rb')
lineset_train = pickle.load(list_file1)
list_file1t = open('data/pkl/test/linepo_{}_{}_{}.pickle'.format(project,version,'test'), 'rb')
lineset_test = pickle.load(list_file1t)
list_file2 = open('data/pkl/test/treetoken_num_{}_{}_{}.pickle'.format(project,version,'train'), 'rb')
treetokennum = pickle.load(list_file2)
list_file2t = open('data/pkl/test/treetoken_num_{}_{}_{}.pickle'.format(project,version,'test'), 'rb')
treetokennum = pickle.load(list_file2t)
list_file3 = open('data/pkl/test/buggy_lines_{}_{}_{}.pickle'.format(project,version,'train'), 'rb')
bug_line_train = pickle.load(list_file3)
list_file3t = open('data/pkl/test/buggy_lines_{}_{}_{}.pickle'.format(project,version,'test'), 'rb')
bug_line_test = pickle.load(list_file3t)
# print(len(new_features_test),len(lineset_test))

def fusionN(project,version,att,lineset,label,new_featu):
    curnum=0
    treenum=0
    temp_1=set()
    temp_pr={}
    temp_label={}
    data_attention=[]
    allabel=[]
    allposition=[]
    print(len(treetokennum))
    for i in range(0,len(lineset)):
        curnum+=1
        if lineset[i] in temp_1:
            temp_pr[lineset[i]]=(temp_pr[lineset[i]]+new_featu[i])/2
        else:
            if args.cuda:
                temp_label[lineset[i]] = int(label[i].cuda().data.cpu().numpy())
            else:
                temp_label[lineset[i]]=int(label[i].detach().numpy())
            temp_1.add(lineset[i])
            temp_pr[lineset[i]]=new_featu[i]
        if curnum == treetokennum[treenum]:
            treenum += 1
            curnum=0
            print(temp_1)
            for j in temp_1:
                allabel.append(temp_label[j])
                data_attention.append(temp_pr[j])
            allposition.append(temp_1)
            temp_1 = set()
            temp_pr = {}
            temp_label={}
    data_filee = open('data/pkl/test/data_attention{}_{}_{}.pickle'.format(project,version,att), 'wb')
    pickle.dump(data_attention, data_filee)
    data_filee.close()
    data_file2 = open('data/pkl/test/data_label_{}_{}_{}.pickle'.format(project,version,att), 'wb')
    pickle.dump(allabel, data_file2)
    data_file2.close()
    data_file3 = open('data/pkl/test/data_position_{}_{}_{}.pickle'.format(project,version,att), 'wb')
    pickle.dump(allposition, data_file3)
    data_file3.close()
    return data_attention,allabel
data_attention_train,allabel_train=fusionN(project,version,'train',lineset_train,labels_train,new_features_train)
data_attention_test,allabel_test=fusionN(project,version,'test',lineset_test,labels_test,new_features_test)

data_attention_train=torch.tensor(data_attention_train, dtype = torch.float32)
data_attention_test=torch.tensor(data_attention_test, dtype = torch.float32)
# x_resample, y_resample = ClusterCentroids(random_state=0).fit_resample(x_resample,y_resample)
ada = ADASYN(random_state=42)
x_resample, y_resample = ada.fit_resample(data_attention_train, allabel_train)

cls = MLPClassifier(alpha=1, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(100, 100), learning_rate='constant',
       learning_rate_init=0.04, max_iter=20000, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
cross_val_score(cls, x_resample, y_resample, cv=10).mean()
cls.fit(x_resample,y_resample)
pro=cls.predict_proba(data_attention_test)
print(pro)
final_pro = open('data/pkl/test/pro/result_{}-{}.pickle'.format(project, version), 'wb')
pickle.dump(pro, final_pro)
final_pro.close()
