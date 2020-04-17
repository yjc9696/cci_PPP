from dgl.nn.pytorch.conv import GATConv
from torch import nn
import torch.nn.functional as F
import torch

class GAT(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation=F.relu,
                 num_heads=2):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(
            GATConv(in_feats=in_feats,
                    out_feats=n_hidden,
                    num_heads=num_heads,
                    activation=activation)
        )

        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                GATConv(in_feats=n_hidden,
                        out_feats=n_hidden,
                        num_heads=num_heads,
                        activation=activation)
            )
        # output layer
        self.linear1 = nn.Linear(n_hidden*3, n_hidden)
        self.dense1_bn = nn.BatchNorm1d(n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_classes)

    def forward(self, g, h, x1, x2):
        
        for layer in self.layers:
            h = layer(g, h)
            h = h.mean(dim=1)
        h = h.contiguous().view(h.shape[0], -1)  # (N, H, d) -> (N, H*d)
        
        h = self.linear1(torch.cat([h[x1], h[x2], torch.abs(h[x1]-h[x2])], 1))
        # h_src = self.linear1(h[x1] + h[x2])
        # mouse doesn't have bn
        # h = self.dense1_bn(h)
        h = F.relu(h)
        h = self.linear2(h)
        return h
