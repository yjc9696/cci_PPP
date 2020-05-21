from dgl.nn.pytorch.conv import GATConv
from torch import nn
import torch.nn.functional as F


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
                GATConv(in_feats=in_feats,
                        out_feats=n_hidden,
                        num_heads=num_heads,
                        activation=activation)
            )
        # output layer
        self.linear1 = nn.Linear(n_hidden * num_heads, n_hidden // 2)
        self.linear2 = nn.Linear(n_hidden // 2, n_classes)

    def forward(self, g, h):
        for layer in self.layers:
            h = layer(g, h).flatten(1)
        h = h.contiguous().view(h.shape[0], -1)  # (N, H, d) -> (N, H*d)
        h = self.linear1(h)
        h = F.relu(h)
        h = self.linear2(h)
        return h
