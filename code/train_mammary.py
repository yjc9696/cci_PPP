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
from datasets import load_mouse_mammary_gland, load_tissue, TrainSet
from models import GraphSAGE, GCN, GAT, VAE, mix_rbf_mmd2, FocalLoss
from torchlight import set_seed
import random

class Trainer:
    def __init__(self, params):
        self.params = params
        
        self.device = torch.device('cpu' if self.params.gpu == -1 else f'cuda:{params.gpu}')
        # self.log_dir = get_dump_path(params) 

        self.batch_size = params.batch_size

        self.load_pretrained_model = params.load_pretrained_model
        self.pretrained_model_path = params.pretrained_model_path
        self.save_model_path = params.save_model_path

        # data
        self.num_cells, self.num_genes, self.num_classes, self.graph, self.features, \
            self.graph_test, self.features_test, self.train_dataset, self.train_mask, \
                self.vali_mask, self.test_dataset = load_mouse_mammary_gland(params)
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
        #                  n_hidden=params.hidden_dim,
        #                  n_classes=self.num_classes,
        #                  n_layers=params.n_layers,
        #                  activation=F.relu)
        self.graph.readonly(readonly_state=True)
        self.graph_test.readonly(readonly_state=True)
        self.model.to(self.device)
        self.features = self.features.to(self.device)
        self.features_test = self.features_test.to(self.device)

        self.graph = self.graph_test
        self.features = self.features_test

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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.lr)
        loss_fn = nn.CrossEntropyLoss(weight=self.loss_weight)
        # loss_fn = FocalLoss()

        ll_loss = 1e5
        
        for epoch in range(self.params.n_epochs):
            self.model.train()
            for step, (batch_x1, batch_x2, batch_y) in enumerate(self.train_dataloader):

                logits = self.model(self.graph, self.features, batch_x1, batch_x2)
                # import pdb; pdb.set_trace()
                loss = loss_fn(logits, batch_y)

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
        # import pdb; pdb.set_trace()
        return precision[1], recall[1], loss

    def test(self, test_dataset):
        self.model.eval()
        loss_fn = nn.CrossEntropyLoss(self.loss_weight)
        with torch.no_grad():
            logits = self.model(self.graph_test, self.features_test, test_dataset[:, 0], test_dataset[:, 1])
            loss = loss_fn(logits, test_dataset[:, 2])
        _, indices = torch.max(logits, dim=1)
        # print(len(indices), indices.sum().item())
        # import pdb; pdb.set_trace()
        precision, recall, f1_score, _ = sklearn.metrics.precision_recall_fscore_support(test_dataset[:,2].tolist(), indices.tolist(), labels=[0,1])
        return precision[1], recall[1], loss


if __name__ == '__main__':
    """
    python ./code/train_mammary.py --dataset 1189 --tissue small_intestine
    python ./code/train_mammary.py --train_dataset 2466 --test_dataset 135 283 352 658 3201 --tissue Peripheral_blood
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
    parser.add_argument("--hidden_dim", type=int, default=200,
                        help="number of hidden gcn units")
    # parser.add_argument("--n_classes", type=int, default=10,
    #                     help="number of classes")
    parser.add_argument("--n_layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--loss_weight", nargs="+", type=int, default=[1, 1],
                        help="loss weight")
    parser.add_argument("--aggregator_type", type=str, default="gcn",
                        help="Aggregator type: mean/gcn/pool/lstm")
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
    parser.add_argument("--train_dataset", type=str, default="test_dataset",
                        help="train dataset")
    parser.add_argument("--test_dataset", type=str, default="train_dataset",
                        help="test dataset")
    parser.add_argument("--just_train", type=int, default=0,
                        help="nothing, for debug")
    parser.add_argument("--each_dataset_size", type=int, default=0,
                        help="0 represent all")
    parser.add_argument("--ligand_receptor_gene", type=str, default='mouse_ligand_receptor_pair.csv',
                        help="cluster - cluster interaction depleted")
    parser.add_argument("--data_dir", type=str, default='mouse_small_intestine',
                        help="root path of the data dir")
    parser.add_argument("--cell_data_path", type=str, default='mouse_small_intestine_1189_data.csv',
                        help="cell data gene")
                        
    parser.add_argument("--cluster_cluster_interaction_enriched", type=str, default='mouse_small_intestine_1189_cluster_cluster_interaction_enriched.csv',
                        help="cluster - cluster interaction enriched")
    parser.add_argument("--cluster_cluster_interaction_depleted", type=str, default='mouse_small_intestine_1189_cluster_cluster_interaction_depleted.csv',
                        help="cluster - cluster interaction depleted")
    parser.add_argument("--cell_cluster", type=str, default='mouse_small_intestine_1189_cellcluster.csv',
                        help="cell belongs to which cluster")
    params = parser.parse_args()
    print(params)
    
    set_seed(params.random_seed)
    # print(random.random())
    # print(np.random.random())
    # print(torch.rand(2))

    trainer = Trainer(params)
    trainer.train()
