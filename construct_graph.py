import os

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch.nn import Parameter, init
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import scipy.sparse as sp
from get_label import get_label
from select_sub import select_sub


def read_data(features, adjs, Y):
    features = np.load(features)
    #print(features.shape)
    features = torch.from_numpy(features)
    features = features.to(torch.float32)
    adjs = np.load(adjs)
    #print(adjs.shape)
    Y = np.load(Y)
    Y = torch.from_numpy(Y)
    Y = Y.to(torch.long)
    graph_dataset = []
    for i in range(len(adjs)):
        adj = adjs[i]
        adj = sp.coo_matrix(adj)
        values = adj.data[:,np.newaxis]
        indices = np.vstack((adj.row, adj.col))  
        adj = torch.LongTensor(indices)
        #values = torch.FloatTensor(values)
        values = torch.from_numpy(values)
        values = values.to(torch.float32)
        #print(adj)
        # adjs[i] = adj
        graph = Data(x=features[i], edge_index=adj, y=Y[i],edge_attr=values)
        # print(graph)
        graph_dataset.append(graph)
    return graph_dataset

def get_conn(datadir):
    subjects_path = os.listdir(datadir)
    subjects_path.sort()
    datapath = []
    feature = []
    for file in subjects_path:
        datapath.append(os.path.join(datadir, file))
    for subject in datapath:
        mat = np.loadtxt(subject)
        feature.append(mat)
    return np.array(feature)

class RESTDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(RESTDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['feature_z_200.npy', '/cc200_0.4.npy', 'Y_label.npy']
    
    @property
    def processed_file_names(self):
        return ['rest-data.pt']


   
    def process(self):
        # Read data into huge `Data` list.
        data_list = read_data(self.raw_dir+'/feature_z_200.npy',self.raw_dir+'/cc200_0.4.npy',self.raw_dir+'/Y_label.npy')

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
#dataset = RESTDataset('dataset/rest-meta-mdd')
#print(dataset.num_classes)
#dataset = TUDataset(os.path.join('data','DD'),name='DD')
#print((dataset[0]))

#graph_dataset = read_data('feature_z_90.npy','adjs_0.2.npy','Y_label.npy')
#print(graph_dataset)
#torch.save(graph_dataset,'rest-mdd-data.pt')
'''
X = get_conn("global_GretnaSFCMatrixZ_Craddock200")
#X_1 = get_conn("extracted/GretnaSFCMatrixR")
y = get_label()
Y = np.reshape(y,[-1,1])
print(Y.shape)
ex_list = select_sub()
X = np.delete(X,ex_list,axis=0)
#X_1 = np.delete(X_1,ex_list,axis=0)
Y = np.delete(Y,ex_list,axis=0)
abs_x = map(abs, X)
adjs = np.array(list(abs_x))
#print(adjs)
adjs = np.where(adjs>0,1,0)
features = X
np.save('feature_z_200.npy',features)
np.save('CC200_adjs_0.npy',adjs)
#np.save('Y_label.npy',Y)
#print(adjs)
graph_dataset = []

#feat = np.load('feature_z_200.npy')
#print(feat.shape)
X = np.load("dataset/rest-meta-mdd/raw/feature_z_200.npy",allow_pickle=True)
abs_x = map(abs, X)
adjs_ori = np.array(list(abs_x))
#print(adjs)
vallist = [0.1,0.2,0.3,0.4,0.5]
actlist = [0.100335347731076, 0.202732554054082, 0.309519604203112, 0.423648930193602, 0.549306144334055]
for i in range(len(vallist)):
    adjs = np.where(adjs_ori>actlist[i],adjs_ori,0.0)
    np.save('dataset/rest-meta-mdd/raw/feature_z_200_'+str(vallist[i])+'.npy',adjs)
'''
'''
#X = np.load("smrir.npy",allow_pickle=True)
#print(X)
y = get_label()
Y = np.reshape(y,[-1,1])
#print(Y.shape)
ex_list = select_sub()
X = np.delete(X,ex_list,axis=0)
#X_1 = np.delete(X_1,ex_list,axis=0)
Y = np.delete(Y,ex_list,axis=0)
abs_x = map(abs, X)
adjs = np.array(list(abs_x))
#print(adjs)
adjs = np.where(adjs>0,1,0)
#features = X
#np.save('feature_z_90.npy',features)
#np.save('smri_adjs_0.npy',adjs)
#np.save('Y_label.npy',Y)
#print(adjs)
graph_dataset = []
'''
#print(1-1e-16)
'''
for i in range(len(adjs)):
    adj = adjs[i]
    adj = sp.coo_matrix(adj)
    values = adj.data
    indices = np.vstack((adj.row, adj.col))  
    adj = torch.LongTensor(indices)
    #print(adj)
    #adjs[i] = adj
    graph = Data(x=features[i],edge_index=adj,y=Y[i])
    #print(graph)
    graph_dataset.append(graph)
print(graph_dataset)
'''
'''
tmp_coo = sp.coo_matrix(adjs)
values = tmp_coo.data
indices = np.vstack((tmp_coo.row,tmp_coo.col))
i = torch.LongTensor(indices)
v = torch.LongTensor(values)
edge_index=torch.sparse_coo_tensor(i,v,tmp_coo.shape)
features = torch.tensor(features, dtype=torch.float)
Y = torch.tensor(Y,dtype=torch.float)
'''
