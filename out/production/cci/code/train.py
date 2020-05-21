import argparse
import time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
# self-defined
from datasets import load_mouse_mammary_gland, load_tissue
from models import GraphSAGE


class Trainer:
    def __init__(self, params):
        self.params = params
        self.train_device = torch.device('cpu' if params.use_cpu else 'cuda:0')
        self.test_device = torch.device('cpu' if params.use_cpu else 'cuda:0')
        # self.log_dir = get_dump_path(params) 

        # data
        self.num_cells, self.num_genes, self.graph, self.features, self.labels, self.train_mask, self.test_mask = load_tissue(params)
        # model
        self.model = GraphSAGE(self.graph, 
                               in_feats=params.dense_dim,
                               n_hidden=params.hidden_dim,
                               n_classes=params.n_classes,
                               n_layers=params.n_layers,
                               activation=F.relu,
                               dropout=params.dropout,
                               aggregator_type=params.aggregator_type)
        
    def train(self):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(params.n_epochs):
            # forward
            self.model.to(self.train_device)
            self.features = self.features.to(self.train_device)
            self.train_mask = self.train_mask.to(self.train_device)
            self.test_mask = self.test_mask.to(self.train_device)
            self.labels = self.labels.to(self.train_device)

            logits = self.model(self.features)
            loss = loss_fn(logits[self.num_genes:][self.train_mask], self.labels[self.train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # acc = self.evaluate(self.train_mask)
            # print("Train Accuracy {:.4f}".format(acc))
            _, _, train_acc = self.evaluate(self.train_mask)
            c, t, test_acc = self.evaluate(self.test_mask)
            if epoch % 20 == 0:
                print(f"Epoch {epoch:04d}: Acc {train_acc:.4f} / {test_acc:.4f}, Loss {loss:.4f}, [{c}/{t}]")


    def evaluate(self, mask):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.features)
            logits = logits[self.num_genes:][mask]
            labels = self.labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels).item()
        total = mask.type(torch.LongTensor).sum().item()
        return correct, total, correct / total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphSAGE')
    parser.add_argument("--random_seed", type=int, default=10086)
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="dropout probability")
    parser.add_argument("--use_cpu", type=bool, default=False)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--n_epochs", type=int, default=500,
                        help="number of training epochs")
    parser.add_argument("--dense_dim", type=int, default=400,
                        help="number of hidden gcn units")
    parser.add_argument("--hidden_dim", type=int, default=200,
                        help="number of hidden gcn units")
    parser.add_argument("--n_classes", type=int, default=10,
                        help="number of classes")
    parser.add_argument("--n_layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--aggregator_type", type=str, default="gcn",
                        help="Aggregator type: mean/gcn/pool/lstm")
    params = parser.parse_args()
    print(params)

    trainer = Trainer(params)
    trainer.train()
