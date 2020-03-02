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
# self-defined
from datasets import load_mouse_mammary_gland, load_tissue, TrainSet
from models import GraphSAGE, GCN, GAT, VAE



class Trainer:
    def __init__(self, params):
        self.params = params
        self.device = torch.device('cpu' if self.params.gpu == -1 else f'cuda:{params.gpu}')
        # self.log_dir = get_dump_path(params) 

        # data
        self.num_cells, self.num_genes, self.num_classes, self.graph, self.features, self.dataset, \
        self.train_mask, self.test_mask, = load_mouse_mammary_gland(params)
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
        self.test_mask = self.test_mask.to(self.device)
        self.dataset = self.dataset.to(self.device)
        self.trainset = TrainSet(self.dataset[self.train_mask])
        self.dataloader = DataLoader(self.trainset, batch_size=8, shuffle=True)



    def train(self):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.lr)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(self.params.n_epochs):
            # forward
            # import pdb; pdb.set_trace()
            for step, (batch_x1, batch_x2, batch_y) in enumerate(self.dataloader):
       
                logits = self.model(self.graph, self.features, batch_x1, batch_x2)
                loss = loss_fn(logits, batch_y)
                # print(logits)
                # print(batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # import pdb; pdb.set_trace()
                # acc = self.evaluate(self.train_mask)
                # print("Train Accuracy {:.4f}".format(acc))
                _, _, train_acc = self.evaluate(self.train_mask)

                c, t, test_acc = self.evaluate(self.test_mask)
                if step % 20 == 0:
                    print(f"Epoch {epoch:04d} Step {step:04d}: Acc {train_acc:.4f} / {test_acc:.4f}, Loss {loss:.4f}, [{c}/{t}]")

            if epoch % 20 == 0:
                print(
                    f"Epoch {epoch:04d}: Acc {train_acc:.4f} / {test_acc:.4f}, Loss {loss:.4f}, [{c}/{t}]")

    def evaluate(self, mask):
        self.model.eval()
        eval_dataset = self.dataset[mask]
        with torch.no_grad():
            logits = self.model(self.graph, self.features, eval_dataset[:, 0], eval_dataset[:, 1])

        _, indices = torch.max(logits, dim=1)
        # import pdb; pdb.set_trace()
        correct = torch.sum(indices == eval_dataset[:,2]).item()
        total = mask.type(torch.LongTensor).sum().item()
        return correct, total, correct / total


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
    parser.add_argument("--aggregator_type", type=str, default="gcn",
                        help="Aggregator type: mean/gcn/pool/lstm")
    # parser.add_argument("--root", type=str, default="../data/mammary_gland",
    #                     help="root path")
    parser.add_argument("--dataset", nargs="+", required=True, type=int,
                        help="list of dataset id")
    parser.add_argument("--tissue", required=True, type=str,
                        help="list of dataset id")

    params = parser.parse_args()
    print(params)

    trainer = Trainer(params)
    trainer.train()
