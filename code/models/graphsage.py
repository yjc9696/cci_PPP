from dgl.nn.pytorch.conv import SAGEConv
from torch import nn
import torch.nn.functional as F
import torch

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
        self.dense1_bn = nn.BatchNorm1d(n_hidden)
        self.linear2 = nn.Linear(n_hidden*3, n_classes)

    def forward(self, g, h, x1, x2):
        for layer in self.layers:
            h = layer(g, h)

        h = self.dense1_bn(h)
        h = self.linear2(torch.cat([h[x1], h[x2], torch.abs(h[x1]-h[x2])], 1))
        # h = self.softmax(h)
        # import pdb;pdb.set_trace()
        # h = F.relu(h)
        # h = self.linear2(h)
        return h

