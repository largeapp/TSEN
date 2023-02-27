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
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GlobalAttention as GA
import pandas as pd
from torch.nn.modules.transformer import TransformerEncoderLayer
from torch.nn.modules.normalization import LayerNorm
from torch.nn.utils.rnn import pad_sequence

device = 'cpu'
seed = 777
batch_size = 16
lr = 1
weight_decay = 0.0001
nhid = 32
pooling_ratio = 0.5
dropout_ratio = 0.6
epochs = 500
patience = 30

class Net(torch.nn.Module):
    def __init__(self,num_classes):
        super(Net,self).__init__()
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



if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    device = 'cuda:0'
#for 10-fold validation
Y = np.zeros((4110,))
features = Y

dataset = TUDataset('dataset','COX2')
num_classes = 2
num_features = 35
loss_func = torch.nn.CrossEntropyLoss()


def model_test(model,loader):
    model.eval()
    correct = 0.
    loss = 0.
    predlist = []
    predprob = []
    gt = []
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.max(dim=1)[1]
        prednp = out.max(dim=1)[1].cpu().numpy()
        predprob.append(out.data.cpu().numpy()[:,1])
        predlist.append(prednp[0])
        gt.append(data.y.cpu().numpy()[0])
        correct += pred.eq(data.y).sum().item()
        loss += loss_func(out,data.y).item()
    return correct / len(loader.dataset),loss / len(loader.dataset),predlist,predprob,gt
balanced_result = []
accuracy_result = []
sensitivity_result = []
specificity_result = []
auc_result = []
f1 = []
tn_list = []
fp_list = []
fn_list = []
tp_list = []

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
k = 0
for train_index, test_index in skf.split(features, Y):
    training_set = dataset[list(train_index)]
    test_set = dataset[list(test_index)]
    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set,batch_size=1,shuffle=False)
    model = Net(num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = WarmupDecayLR(optimizer, warmup_steps=10000, d_model=35)
    scaler = GradScaler()
    min_loss = 1e10
    patience = 0
    max_acc = 0
    for epoch in range(epochs):
        model.train()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(device)
            out = model(data)
            loss = loss_func(out, data.y)
            print("Training loss:{}".format(loss.item()))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        val_acc,val_loss,_,_,_ = model_test(model,test_loader)
        print("Current round:" + str(k))
        print("Epoch {}".format(epoch))
        print("Validation loss:{}\taccuracy:{}".format(val_loss,val_acc))
        print("Current max val acc:"+str(max_acc))
        if val_acc > max_acc:
            torch.save(model.state_dict(),'sten_latest'+str(k)+'.pth')
            print("Model saved at epoch{}".format(epoch))
            max_acc = val_acc
            patience = 0
        else:
            patience += 1
        if patience > patience:
            break
    del model

    model = Net(num_classes).to(device)
    model.load_state_dict(torch.load('sten_latest'+str(k)+'.pth'))
    test_acc,test_loss,predlist,predprob,gt  = model_test(model,test_loader)
    print("Test accuarcy:{}".format(test_acc))
    y_pred = np.array(predlist)
    y_predprob = np.array(predprob)
    y_test = np.array(gt)
    balanced_result.append(balanced_accuracy_score(y_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    tn_list.append(tn)
    fp_list.append(fp)
    fn_list.append(fn)
    tp_list.append(tp)
    sensitivity_result.append(tp / (tp + fn))
    specificity_result.append(tn / (fp + tn))
    accuracy_result.append(accuracy_score(y_test.data, y_pred))
    auc_result.append(roc_auc_score(y_test.data, y_predprob))
    f1.append(f1_score(y_test, y_pred, zero_division=1))
    k += 1

print("average accuracy: " + str(np.mean(accuracy_result)))
print("var:" + str(np.var(accuracy_result)))
print("max:" + str(np.max(accuracy_result)))
print("min:" + str(np.min(accuracy_result)))
print("average balanced_accuracy: " + str(np.mean(balanced_result)))
print("var:" + str(np.var(balanced_result)))
print("max:" + str(np.max(balanced_result)))
print("min:" + str(np.min(balanced_result)))
print("average sensitivity: " + str(np.mean(sensitivity_result)))
print("var:" + str(np.var(sensitivity_result)))
print("max:" + str(np.max(sensitivity_result)))
print("min:" + str(np.min(sensitivity_result)))
print("average specificity: " + str(np.mean(specificity_result)))
print("var:" + str(np.var(specificity_result)))
print("max:" + str(np.max(specificity_result)))
print("min:" + str(np.min(specificity_result)))
print("auc accuracy: " + str(np.mean(auc_result)))
print("var:" + str(np.var(auc_result)))
print("max:" + str(np.max(auc_result)))
print("min:" + str(np.min(auc_result)))
print("average f1 score:" + str(np.mean(f1)))
print("var:" + str(np.var(f1)))
print("max:" + str(np.max(f1)))
print("min:" + str(np.min(f1)))
print("average tn:" + str(np.mean(tn_list)))
print("average fp:" + str(np.mean(fp_list)))
print("average fn:" + str(np.mean(fn_list)))
print("average tp:" + str(np.mean(tp_list)))
final_result = [(np.mean(accuracy_result),np.var(accuracy_result),np.mean(balanced_result),np.var(balanced_result),np.mean(sensitivity_result),np.var(sensitivity_result),
          np.mean(specificity_result),np.var(specificity_result),np.mean(auc_result),np.var(auc_result),np.mean(f1),np.var(f1),np.mean(tn_list),np.mean(fp_list),
          np.mean(fn_list),np.mean(tp_list))]
analyse_table = pd.DataFrame(final_result, columns=['accuracy','acc var', 'balanced accuracy', 'balanced acc var', 'sensitivity', 'sensitivity var',
                                                   'specificity', 'specificity var','auc', 'auc var','f1', 'f1 var',
                                                   'tn', 'fp','fn','tp'])
analyse_table.to_csv('exp_result_sten.csv')