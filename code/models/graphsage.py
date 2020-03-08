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

        # self.dropout = nn.Dropout(p=0.5)

        # output layer
        self.linear1 = nn.Linear(n_hidden*3, n_hidden)
        self.dense1_bn = nn.BatchNorm1d(n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_classes)

    def forward(self, g, h, x1, x2, x1_tar, x2_tar):
        for layer in self.layers:
            h = layer(g, h)

        # h = self.linear1(h)
        
        h_src = self.linear1(torch.cat([h[x1], h[x2], torch.abs(h[x1]-h[x2])], 1))
        h_tar = self.linear1(torch.cat([h[x1_tar], h[x2_tar], torch.abs(h[x1_tar]-h[x2_tar])], 1))
        # mouse doesn't have bn
        h_src = self.dense1_bn(h_src)
        h_tar = self.dense1_bn(h_tar)

        h_src_mmd = F.relu(h_src)
        h_tar_mmd = F.relu(h_tar)

        h_p = self.linear2(h_src_mmd)

        return h_p, h_src_mmd, h_tar_mmd
        # return h, 0, 0

