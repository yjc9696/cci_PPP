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
# self-defined
from datasets import load_mouse_mammary_gland, load_tissue, TrainSet
from models import GraphSAGE, GCN, GAT, VAE



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
        self.num_cells, self.num_genes, self.num_classes, self.graph, self.features, self.train_dataset, \
        self.train_mask, self.vali_mask, self.test_dataset = load_mouse_mammary_gland(params)
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
        self.model.to(self.device)
        self.features = self.features.to(self.device)
        self.train_mask = self.train_mask.to(self.device)
        self.vali_mask = self.vali_mask.to(self.device)
        self.train_dataset = self.train_dataset.to(self.device)
        self.trainset = TrainSet(self.train_dataset[self.train_mask])
        self.test_dataset = self.test_dataset.to(self.device)
        self.dataloader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
        self.loss_weight = torch.Tensor([1, params.loss_weight]).to(self.device)

    def train(self):
        if self.load_pretrained_model:
            self.model.load_state_dict(torch.load(self.pretrained_model_path))
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.lr)
        loss_fn = nn.CrossEntropyLoss(weight=self.loss_weight)

        ll_loss = 1e5
        for epoch in range(self.params.n_epochs):
            for step, (batch_x1, batch_x2, batch_y) in enumerate(self.dataloader):
       
                logits = self.model(self.graph, self.features, batch_x1, batch_x2)
                # print(logits)
                loss = loss_fn(logits, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                _, _, train_loss = self.evaluate(self.train_mask)
                precision, recall, vali_loss = self.evaluate(self.vali_mask)
                
            if (ll_loss - vali_loss) / ll_loss > 0.005:
                torch.save(self.model.state_dict(), self.save_model_path)

            if epoch % 1 == 0:
                precision, recall, train_loss = self.evaluate(self.train_mask)
                print(f"Epoch {epoch:04d}: precesion {precision:.5f}, recall {recall:05f}, train loss: {vali_loss}")
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
        # print(eval_dataset[:, 2][:100])
        # print(indices[:100])
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
    parser.add_argument("--loss_weight", type=int, default=1,
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
    parser.add_argument("--train_dataset", type=str, default="train_dataset2",
                        help="train dataset")
    parser.add_argument("--test_dataset", type=str, default="test_dataset2",
                        help="test dataset")                    
    params = parser.parse_args()
    print(params)

    trainer = Trainer(params)
    trainer.train()
