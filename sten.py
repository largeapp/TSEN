from collections import Counter
import os

from torch_geometric.data import DataLoader
from scheduler import WarmupDecayLR
from torch.cuda.amp import GradScaler
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch_geometric.nn import GlobalAttention as GA
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv
from construct_graph_for_pyg import RESTDataset
from torch_geometric.nn import GlobalAttention as GA
import pandas as pd
from torch.nn.modules.transformer import TransformerEncoderLayer
from torch.nn.modules.normalization import LayerNorm
from torch.nn.utils.rnn import pad_sequence


class STENabide(torch.nn.Module):
    def __init__(self,num_classes):
        super(STENabide,self).__init__()
        self.gcn1 = GCNConv(200,200)
        self.gcn2 = GCNConv(400,200)
        self.num_classes = num_classes
        self.ga1 = GA(torch.nn.Sequential(torch.nn.Linear(400,128),torch.nn.Linear(128,1)))
        self.ga2 = GA(torch.nn.Sequential(torch.nn.Linear(600,200),torch.nn.Linear(200,1)))
        self.l1 = torch.nn.Linear(1000,128)#352
        self.l2 = torch.nn.Linear(128,32)
        self.l3 = torch.nn.Linear(32,num_classes)
        self.b1 = torch.nn.BatchNorm1d(num_features=128)
        self.b2 = torch.nn.BatchNorm1d(num_features=32)
        self.self_attn1 = TransformerEncoderLayer(400,5,1024,0.1,"gelu")
        self.self_attn2 = TransformerEncoderLayer(600,5,1024,0.1,"gelu")
        torch.nn.init.kaiming_normal_(self.l1.weight, a=1)
        torch.nn.init.kaiming_normal_(self.l2.weight, a=1)
        torch.nn.init.kaiming_normal_(self.l3.weight, a=1)
    def _graph2seq(self,x,batch):
        batch_list = list(batch.cpu().numpy())
        counter = Counter(batch_list)
        count = 0
        x_list = []
        for i in range(batch_list[-1]+1):
            seq = x[count:count+counter[i]]
            count += counter[i]
            x_list.append(seq)
        seq = pad_sequence(x_list,padding_value=1e9)
        seq_padding_mask = (seq == 1e9)[:,:,0].T
        return seq,seq_padding_mask
    
    def _seq2graph(self,seq,mask):
        seq = seq.transpose(1,0)
        x_list = []
        for i in range(seq.shape[0]):
            for j in range(seq.shape[1]):
                if not mask[i][j]:
                    x_list.append(seq[i][j])
        return torch.stack(x_list,dim=0)
    
    def forward(self,data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        res = x #feature for layer 0
        
        #first layer of Snowball Encoding layer
        x = self.gcn1(x,edge_index)
        x = F.leaky_relu(x)
        x_1 = F.dropout(x,0.5,training=self.training)#feature for layer 1
        x_1,x_1_mask = self._graph2seq(x_1,batch)
        emb = x_1.shape[0]
        res,res_mask = self._graph2seq(res,batch)
        x = torch.cat([x_1, res], dim=2)
        x_mask = res_mask
        x = self.self_attn1(src=x,src_key_padding_mask = x_mask)
        res_seq = x[:,:,emb:]
        x_1_seq = x[:,:,:emb]    
        res = self._seq2graph(res_seq,res_mask)
        x_1 = self._seq2graph(x_1_seq,x_1_mask)
        x = torch.cat([x_1, res], dim=1)

        x1 = self.ga1(x,batch)#graph representation
        
        #second layer of snowball encoding layer
        x  = self.gcn2(x,edge_index)
        x = F.leaky_relu(x)
        x_2 = F.dropout(x, 0.5, training=self.training)       
        x_2,x_2_mask = self._graph2seq(x_2,batch)
        x = torch.cat([x_2,x_1_seq,res_seq],dim=2)
        x = self.self_attn2(src=x,src_key_padding_mask = x_mask)
        x_2_seq = x[:,:,:emb]
        x_1_seq = x[:,:,emb:emb*2]
        res_seq = x[:,:,emb*2:]
        x_2 = self._seq2graph(x_2_seq,x_2_mask)
        x_1 = self._seq2graph(x_1_seq,x_1_mask)
        res = self._seq2graph(res_seq,res_mask)
        x = torch.cat([x_2, x_1, res], dim=1)        

        x2 = self.ga2(x,batch)

        x = torch.cat([x1, x2], dim=1)

        x = self.l1(x)
        x = self.b1(x)
        x = F.leaky_relu(x)
        x = F.dropout(x,0.5, training=self.training)

        x = self.l2(x)
        x = self.b2(x)
        x = F.leaky_relu(x)
        x = F.dropout(x,0.5, training=self.training)

        x = self.l3(x)

        return F.softmax(x)


class STENmdd(torch.nn.Module):
    def __init__(self,num_classes):
        super(STENmdd,self).__init__()
        self.gcn1 = GCNConv(200,200)
        self.gcn2 = GCNConv(400,200)
        self.num_classes = num_classes
        self.ga1 = GA(torch.nn.Sequential(torch.nn.Linear(400,128),torch.nn.Linear(128,1)))
        self.ga2 = GA(torch.nn.Sequential(torch.nn.Linear(600,200),torch.nn.Linear(200,1)))
        self.l1 = torch.nn.Linear(1000,256)#352
        self.l2 = torch.nn.Linear(256,64)
        self.l3 = torch.nn.Linear(64,num_classes)
        self.b1 = torch.nn.BatchNorm1d(num_features=256)
        self.b2 = torch.nn.BatchNorm1d(num_features=64)
        self.self_attn1 = TransformerEncoderLayer(400,5,1024,0.1,"gelu")
        self.self_attn2 = TransformerEncoderLayer(600,5,1024,0.1,"gelu")
        torch.nn.init.kaiming_normal_(self.l1.weight, a=1)
        torch.nn.init.kaiming_normal_(self.l2.weight, a=1)
        torch.nn.init.kaiming_normal_(self.l3.weight, a=1)
    def _graph2seq(self,x,batch):
        batch_list = list(batch.cpu().numpy())
        counter = Counter(batch_list)
        count = 0
        x_list = []
        for i in range(batch_list[-1]+1):
            seq = x[count:count+counter[i]]
            count += counter[i]
            x_list.append(seq)
        seq = pad_sequence(x_list,padding_value=1e9)
        seq_padding_mask = (seq == 1e9)[:,:,0].T
        return seq,seq_padding_mask
    
    def _seq2graph(self,seq,mask):
        seq = seq.transpose(1,0)
        x_list = []
        for i in range(seq.shape[0]):
            for j in range(seq.shape[1]):
                if not mask[i][j]:
                    x_list.append(seq[i][j])
        return torch.stack(x_list,dim=0)
    
    def forward(self,data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        res = x #feature for layer 0
        
        #first layer of Snowball Encoding layer
        x = self.gcn1(x,edge_index)
        x = F.leaky_relu(x)
        x_1 = F.dropout(x,0.5,training=self.training)#feature for layer 1
        x_1,x_1_mask = self._graph2seq(x_1,batch)
        emb = x_1.shape[0]
        res,res_mask = self._graph2seq(res,batch)
        x = torch.cat([x_1, res], dim=2)
        x_mask = res_mask
        x = self.self_attn1(src=x,src_key_padding_mask = x_mask)
        res_seq = x[:,:,emb:]
        x_1_seq = x[:,:,:emb]    
        res = self._seq2graph(res_seq,res_mask)
        x_1 = self._seq2graph(x_1_seq,x_1_mask)
        x = torch.cat([x_1, res], dim=1)

        x1 = self.ga1(x,batch)#graph representation
        
        #second layer of snowball encoding layer
        x  = self.gcn2(x,edge_index)
        x = F.leaky_relu(x)
        x_2 = F.dropout(x, 0.5, training=self.training)       
        x_2,x_2_mask = self._graph2seq(x_2,batch)
        x = torch.cat([x_2,x_1_seq,res_seq],dim=2)
        x = self.self_attn2(src=x,src_key_padding_mask = x_mask)
        x_2_seq = x[:,:,:emb]
        x_1_seq = x[:,:,emb:emb*2]
        res_seq = x[:,:,emb*2:]
        x_2 = self._seq2graph(x_2_seq,x_2_mask)
        x_1 = self._seq2graph(x_1_seq,x_1_mask)
        res = self._seq2graph(res_seq,res_mask)
        x = torch.cat([x_2, x_1, res], dim=1)        

        x2 = self.ga2(x,batch)

        x = torch.cat([x1, x2], dim=1)

        x = self.l1(x)
        x = self.b1(x)
        x = F.leaky_relu(x)
        x = F.dropout(x,0.5, training=self.training)

        x = self.l2(x)
        x = self.b2(x)
        x = F.leaky_relu(x)
        x = F.dropout(x,0.5, training=self.training)

        x = self.l3(x)

        return F.softmax(x)

class STENcox2(torch.nn.Module):
    def __init__(self,num_classes):
        super(STENcox2,self).__init__()
        self.gcn1 = GCNConv(35,35)
        self.gcn2 = GCNConv(70,35)
        self.num_classes = num_classes
        self.ga1 = GA(torch.nn.Sequential(torch.nn.Linear(70,16),torch.nn.Linear(16,1)))
        self.ga2 = GA(torch.nn.Sequential(torch.nn.Linear(105,35),torch.nn.Linear(35,1)))
        self.l1 = torch.nn.Linear(175,64)#352
        self.l2 = torch.nn.Linear(64,16)
        self.l3 = torch.nn.Linear(16,num_classes)
        self.b1 = torch.nn.BatchNorm1d(num_features=64)
        self.b2 = torch.nn.BatchNorm1d(num_features=16)
        self.self_attn1 = TransformerEncoderLayer(70,5,1024,0.1,"gelu")
        self.self_attn2 = TransformerEncoderLayer(105,5,1024,0.1,"gelu")
        torch.nn.init.kaiming_normal_(self.l1.weight, a=1)
        torch.nn.init.kaiming_normal_(self.l2.weight, a=1)
        torch.nn.init.kaiming_normal_(self.l3.weight, a=1)
    def _graph2seq(self,x,batch):
        batch_list = list(batch.cpu().numpy())
        counter = Counter(batch_list)
        count = 0
        x_list = []
        for i in range(batch_list[-1]+1):
            seq = x[count:count+counter[i]]
            count += counter[i]
            x_list.append(seq)
        seq = pad_sequence(x_list,padding_value=1e9)
        seq_padding_mask = (seq == 1e9)[:,:,0].T
        return seq,seq_padding_mask
    
    def _seq2graph(self,seq,mask):
        seq = seq.transpose(1,0)
        x_list = []
        for i in range(seq.shape[0]):
            for j in range(seq.shape[1]):
                if not mask[i][j]:
                    x_list.append(seq[i][j])
        return torch.stack(x_list,dim=0)
    
    def forward(self,data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        res = x #feature for layer 0
        
        #first layer of Snowball Encoding layer
        x = self.gcn1(x,edge_index)
        x = F.leaky_relu(x)
        x_1 = F.dropout(x,0.5,training=self.training)#feature for layer 1
        x_1,x_1_mask = self._graph2seq(x_1,batch)
        emb = x_1.shape[0]
        res,res_mask = self._graph2seq(res,batch)
        x = torch.cat([x_1, res], dim=2)
        x_mask = res_mask
        x = self.self_attn1(src=x,src_key_padding_mask = x_mask)
        res_seq = x[:,:,emb:]
        x_1_seq = x[:,:,:emb]    
        res = self._seq2graph(res_seq,res_mask)
        x_1 = self._seq2graph(x_1_seq,x_1_mask)
        x = torch.cat([x_1, res], dim=1)

        x1 = self.ga1(x,batch)#graph representation
        
        #second layer of snowball encoding layer
        x  = self.gcn2(x,edge_index)
        x = F.leaky_relu(x)
        x_2 = F.dropout(x, 0.5, training=self.training)       
        x_2,x_2_mask = self._graph2seq(x_2,batch)
        x = torch.cat([x_2,x_1_seq,res_seq],dim=2)
        x = self.self_attn2(src=x,src_key_padding_mask = x_mask)
        x_2_seq = x[:,:,:emb]
        x_1_seq = x[:,:,emb:emb*2]
        res_seq = x[:,:,emb*2:]
        x_2 = self._seq2graph(x_2_seq,x_2_mask)
        x_1 = self._seq2graph(x_1_seq,x_1_mask)
        res = self._seq2graph(res_seq,res_mask)
        x = torch.cat([x_2, x_1, res], dim=1)        

        x2 = self.ga2(x,batch)

        x = torch.cat([x1, x2], dim=1)

        x = self.l1(x)
        x = self.b1(x)
        x = F.leaky_relu(x)
        x = F.dropout(x,0.5, training=self.training)

        x = self.l2(x)
        x = self.b2(x)
        x = F.leaky_relu(x)
        x = F.dropout(x,0.5, training=self.training)

        x = self.l3(x)

        return F.softmax(x)

class STENnci(torch.nn.Module):
    def __init__(self,num_classes):
        super(STENnci,self).__init__()
        self.gcn1 = GCNConv(37,37)
        self.gcn2 = GCNConv(74,37)
        self.gcn3 = GCNConv(111,37)
        self.num_classes = num_classes
        self.ga1 = GA(torch.nn.Sequential(torch.nn.Linear(74,8),torch.nn.Linear(8,1)))
        self.ga2 = GA(torch.nn.Sequential(torch.nn.Linear(111,16),torch.nn.Linear(16,1)))   
        self.ga3 = GA(torch.nn.Sequential(torch.nn.Linear(148,16),torch.nn.Linear(16,1)))        
        self.l1 = torch.nn.Linear(333,128)#352
        self.l2 = torch.nn.Linear(128,32)#352
        self.l3 = torch.nn.Linear(32,num_classes)
        self.b1 = torch.nn.BatchNorm1d(num_features=74)
        self.b2 = torch.nn.BatchNorm1d(num_features=16)
        self.b1 = torch.nn.BatchNorm1d(num_features=128)
        self.b2 = torch.nn.BatchNorm1d(num_features=32)
        self.self_attn1 = TransformerEncoderLayer(74,1,256,0.1,"gelu")
        self.self_attn2 = TransformerEncoderLayer(111,1,256,0.1,"gelu")
        self.self_attn3 = TransformerEncoderLayer(148,1,256,0.1,"gelu")
        torch.nn.init.kaiming_normal_(self.l1.weight, a=1)
        torch.nn.init.kaiming_normal_(self.l2.weight, a=1)
        torch.nn.init.kaiming_normal_(self.l3.weight, a=1)
    def _graph2seq(self,x,batch):
        batch_list = list(batch.cpu().numpy())
        counter = Counter(batch_list)
        count = 0
        x_list = []
        for i in range(batch_list[-1]+1):
            seq = x[count:count+counter[i]]
            count += counter[i]
            x_list.append(seq)
        seq = pad_sequence(x_list,padding_value=1e9)
        seq_padding_mask = (seq == 1e9)[:,:,0].T
        return seq,seq_padding_mask
    
    def _seq2graph(self,seq,mask):
        seq = seq.transpose(1,0)
        x_list = []
        for i in range(seq.shape[0]):
            for j in range(seq.shape[1]):
                if not mask[i][j]:
                    x_list.append(seq[i][j])
        return torch.stack(x_list,dim=0)
    
    def forward(self,data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        res = x #feature for layer 0
        
        #first layer of Snowball Encoding layer
        x = self.gcn1(x,edge_index)
        x = F.leaky_relu(x)
        x_1 = F.dropout(x,0.5,training=self.training)#feature for layer 1
        x_1,x_1_mask = self._graph2seq(x_1,batch)
        emb = x_1.shape[0]
        res,res_mask = self._graph2seq(res,batch)
        x = torch.cat([x_1, res], dim=2)
        x_mask = res_mask
        x = self.self_attn1(src=x,src_key_padding_mask = x_mask)
        res_seq = x[:,:,emb:]
        x_1_seq = x[:,:,:emb]    
        res = self._seq2graph(res_seq,res_mask)
        x_1 = self._seq2graph(x_1_seq,x_1_mask)
        x = torch.cat([x_1, res], dim=1)

        x1 = self.ga1(x,batch)#graph representation
        
        #second layer of snowball encoding layer
        x  = self.gcn2(x,edge_index)
        x = F.leaky_relu(x)
        x_2 = F.dropout(x, 0.5, training=self.training)       
        x_2,x_2_mask = self._graph2seq(x_2,batch)
        x = torch.cat([x_2,x_1_seq,res_seq],dim=2)
        x = self.self_attn2(src=x,src_key_padding_mask = x_mask)
        x_2_seq = x[:,:,:emb]
        x_1_seq = x[:,:,emb:emb*2]
        res_seq = x[:,:,emb*2:]
        x_2 = self._seq2graph(x_2_seq,x_2_mask)
        x_1 = self._seq2graph(x_1_seq,x_1_mask)
        res = self._seq2graph(res_seq,res_mask)
        x = torch.cat([x_2, x_1, res], dim=1)        

        x2 = self.ga2(x,batch)

        #third layer of snowball encoding layer
        x = self.gcn3(x,edge_index)
        x = F.leaky_relu(x)
        x_3 = F.dropout(x, 0.5, training=self.training)       
        x_3_seq,x_3_mask = self._graph2seq(x_3,batch)
        x = torch.cat([x_3_seq,x_2_seq,x_1_seq,res_seq],dim=2)
        x = self.self_attn3(src=x,src_key_padding_mask = x_mask)
        x_3_seq = x[:,:,:emb]
        x_2_seq = x[:,:,emb:emb*2]
        x_1_seq = x[:,:,emb*2:emb*3]
        res_seq = x[:,:,emb*3:]
        x_3 = self._seq2graph(x_3_seq,x_3_mask)
        x_2 = self._seq2graph(x_2_seq,x_2_mask)
        x_1 = self._seq2graph(x_1_seq,x_1_mask)
        res = self._seq2graph(res_seq,res_mask)
        x = torch.cat([x_3, x_2, x_1, res], dim=1)
        x3 = self.ga3(x,batch)

        x = torch.cat([x1, x2, x3], dim=1)

        x = self.l1(x)
        x = self.b1(x)
        x = F.leaky_relu(x)
        x = F.dropout(x,0.5, training=self.training)

        x = self.l2(x)
        x = self.b2(x)
        x = F.leaky_relu(x)
        x = F.dropout(x,0.5, training=self.training)

        x = self.l3(x)

        return F.softmax(x)

dataset = RESTDataset('dataset/abide')
