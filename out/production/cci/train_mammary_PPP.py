import argparse
import time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dgl
from dgl.contrib.sampling import NeighborSampler
from sklearn.metrics import precision_recall_fscore_support
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
# self-defined
from datasets import load_PPP_mammary_gland, load_tissue, TrainSet
from models import GraphSAGE, GCN, GAT, VAE, mix_rbf_mmd2
from torchlight import set_seed
import random


class Trainer:
    def __init__(self, params):
        self.params = params
        
        self.device = torch.device('cpu' if self.params.gpu == -1 else f'cuda:{params.gpu}')
        print(self.device)
        # self.log_dir = get_dump_path(params) 

        self.batch_size = params.batch_size

        self.load_pretrained_model = params.load_pretrained_model
        self.pretrained_model_path = params.pretrained_model_path
        self.save_model_path = params.save_model_path

        self.using_mmd = params.using_mmd

        # data
        self.num_cells, self.num_genes, self.num_classes, self.graph, self.features, self.train_dataset, \
        self.train_mask, self.vali_mask, self.test_dataset = load_PPP_mammary_gland(params)
        # self.vae = torch.load('./saved_model/vae.pkl', self.features.device)
        # self.features = self.vae.get_hidden(self.features)
        # model
        self.model = GraphSAGE(in_feats=params.dense_dim,
                               n_hidden=params.hidden_dim,
                               n_classes=self.num_classes,
                               n_layers=params.n_layers,
                               activation=F.relu,
                               dropout=params.dropout,
                               aggregator_type=params.aggregator_type,
                               num_genes=self.num_genes)
        # self.model = GCN(
        #                  in_feats=params.dense_dim,
        #                  n_hidden=params.hidden_dim,
        #                  n_classes=self.num_classes,
        #                  n_layers=params.n_layers,
        #                  activation=F.relu)
        # self.model = GAT(in_feats=params.dense_dim,
        #                  n_hidden=100,
        #                  n_classes=self.num_classes,
        #                  n_layers=params.n_layers,
        #                  activation=F.relu)
        self.graph.readonly(readonly_state=True)
        self.loss_fn = nn.BCEWithLogitsLoss().to(self.device)
        self.model.to(self.device)
        self.features = self.features.to(self.device)
        self.train_mask = self.train_mask.to(self.device)
        self.vali_mask = self.vali_mask.to(self.device)
        self.train_dataset = self.train_dataset.to(self.device)
        self.trainset = TrainSet(self.train_dataset[self.train_mask])
        self.test_dataset = self.test_dataset.to(self.device)
        self.train_dataloader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.loss_weight = torch.Tensor(params.loss_weight).to(self.device)

    def train(self):
        if self.load_pretrained_model:
            print(f'load model from {self.pretrained_model_path}')
            self.model.load_state_dict(torch.load(self.pretrained_model_path))
        self.model.train()
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.lr)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.lr, weight_decay=0.0001)

        ll_loss = 1e5
        print("start train")
        for epoch in range(self.params.n_epochs):
            self.model.train()
            for step, (batch_x1, batch_x2, batch_y_classify,batch_y) in enumerate(self.train_dataloader):
                # list_tar = list(enumerate(self.test_dataloader))

                logits = self.model(self.graph, self.features, batch_x1, batch_x2)
                # import pdb; pdb.set_trace()
                loss = self.loss_fn(logits.squeeze_(), batch_y_classify.type(torch.float))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            _, _, train_loss = self.evaluate(self.train_mask)
            precision, recall, vali_loss = self.evaluate(self.vali_mask)
                
            # if vali_loss < ll_loss:
            #     torch.save(self.model.state_dict(), self.save_model_path)
            #     ll_loss = vali_loss
            if train_loss < ll_loss:
                torch.save(self.model.state_dict(), self.save_model_path)
                ll_loss = train_loss

            if epoch % 1 == 0:
                precision, recall, train_loss = self.evaluate(self.train_mask)
                print(f"Epoch {epoch:04d}: precesion {precision:.5f}, recall {recall:05f}, train loss: {vali_loss}")
                if self.params.just_train == 0:
                    precision, recall, vali_loss = self.evaluate(self.vali_mask)
                    print(f"Epoch {epoch:04d}: precesion {precision:.5f}, recall {recall:05f}, vali loss: {vali_loss}")
                    precision, recall, test_loss = self.test(self.test_dataset)
                    print(f"Epoch {epoch:04d}: precesion {precision:.5f}, recall {recall:05f}, test loss: {test_loss}")

    def evaluate(self, mask):
        self.model.eval()
        eval_dataset = self.train_dataset[mask]
        loss_fn = nn.CrossEntropyLoss(self.loss_weight)
        with torch.no_grad():
            logits = self.model(self.graph, self.features, eval_dataset[:, 0], eval_dataset[:, 1])
            loss = loss_fn(logits, eval_dataset[:, 2])
        _, indices = torch.max(logits, dim=1)
        ap_score = average_precision_score(eval_dataset[:,2].tolist(), indices.tolist())
        precision, recall, f1_score, _ = sklearn.metrics.precision_recall_fscore_support(eval_dataset[:,2].tolist(), indices.tolist(), labels=[0,1])
        return precision[1], recall[1], loss

    def test(self, test_dataset):
        self.model.eval()
        eval_dataset = test_dataset
        loss_fn = nn.CrossEntropyLoss(self.loss_weight)
        with torch.no_grad():
            logits = self.model(self.graph, self.features, eval_dataset[:, 0], eval_dataset[:, 1])
            loss = loss_fn(logits, eval_dataset[:, 2])
        _, indices = torch.max(logits, dim=1)
        precision, recall, f1_score, _ = sklearn.metrics.precision_recall_fscore_support(eval_dataset[:,2].tolist(), indices.tolist(), labels=[0,1])
        return precision[1], recall[1], loss

if __name__ == '__main__':
    """
    python ./code/train_mammary_PPP.py --dataset 10000 --tissue blood --dense_dim 2000 --hidden_dim 1000 --n_epochs 100 --train_dataset train_dataset10000 --test_dataset test_dataset10000 --save_model_path checekpoints_ppp_10000.pth
    python ./code/train_mammary_PPP.py --dataset 50000 --tissue blood --dense_dim 2000 --n_epochs 100 --train_dataset train_dataset50000 --test_dataset test_dataset50000 --save_model_path checekpoints_ppp_50000.pth
    python ./code/train_mammary_PPP.py --dataset 350000 --tissue blood --dense_dim 2000 --n_epochs 100 --train_dataset train_dataset_all --test_dataset test_dataset_all --save_model_path checekpoints_ppp_all.pth
    python ./code/train_mammary_PPP.py --lr 1e-3 --dataset 350000 --tissue blood --dense_dim 2000 --n_epochs 100 --train_dataset train_dataset_all_cross --test_dataset test_dataset_all_cross --save_model_path checekpoints_ppp_all_cross.pth
   
    """
    parser = argparse.ArgumentParser(description='GraphSAGE')
    parser.add_argument("--random_seed", type=int, default=10086)
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU id, -1 for cpu")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--n_epochs", type=int, default=500,
                        help="number of training epochs")
    parser.add_argument("--dense_dim", type=int, default=400,
                        help="number of hidden gcn units")
    parser.add_argument("--hidden_dim", type=int, default=2000,
                        help="number of hidden gcn units")
    # parser.add_argument("--n_classes", type=int, default=10,
    #                     help="number of classes")
    parser.add_argument("--n_layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--loss_weight", nargs="+", type=int, default=[1, 1],
                        help="loss weight")
    parser.add_argument("--aggregator_type", type=str, default="gcn",
                        help="Aggregator type: mean/gcn/pool/lstm")
    # parser.add_argument("--root", type=str, default="../data/mammary_gland",
    #                     help="root path")
    parser.add_argument("--dataset", nargs="+", required=True, type=int,
                        help="list of dataset id")
    parser.add_argument("--tissue", required=True, type=str,
                        help="list of dataset id")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="size of batch")
    parser.add_argument("--ligand_receptor_pair_path", type=str, default="mouse_ligand_receptor_pair",
                        help="gene ligand receptor pair path")
    parser.add_argument("--pretrained_model_path", type=str, default="checkpoints_default.pth",
                        help="pretrained_model_path")
    parser.add_argument("--load_pretrained_model", type=int, default=0,
                        help="load_pretrained_model")                                   
    parser.add_argument("--save_model_path", type=str, default="checkpoints_default.pth",
                        help="save_model_path")
    parser.add_argument("--train_dataset", type=str, default="train_dataset",
                        help="train dataset")
    parser.add_argument("--test_dataset", type=str, default="test_dataset",
                        help="test dataset")
    parser.add_argument("--just_train", type=int, default=0,
                        help="nothing, for debug")
    parser.add_argument("--each_dataset_size", type=int, default=0,
                        help="0 represent all")
    parser.add_argument("--using_mmd", type=int, default=0,
                        help="if using mmd loss, 0 is not using")            
    params = parser.parse_args()
    print(params)
    
    set_seed(params.random_seed)
    # print(random.random())
    # print(np.random.random())
    # print(torch.rand(2))

    trainer = Trainer(params)
    trainer.train()
