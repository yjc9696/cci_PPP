from dgl.nn.pytorch.conv import SAGEConv
from torch import nn
import torch.nn.functional as F


class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type,
                 num_genes):
        super(GraphSAGE, self).__init__()

        self.num_genes = num_genes

        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(
                SAGEConv(in_feats,
                         n_hidden, 
                         aggregator_type, 
                         feat_drop=dropout, 
                         activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                    SAGEConv(n_hidden, 
                             n_hidden,
                             aggregator_type, 
                             feat_drop=dropout, 
                             activation=activation))
        # output layer
        # self.linear1 = nn.Linear(n_hidden, n_hidden // 2)
        self.linear2 = nn.Linear(n_hidden, n_classes)

    def forward(self, g, h, x):
        for layer in self.layers:
            h = layer(g, h)
        # change to Cartesian product,

        h = self.linear2(h[x[:, 0]] + h[x[:, 1]])
        # h = F.relu(h)
        # h = self.linear2(h)
        return h

