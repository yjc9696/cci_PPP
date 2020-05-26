import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, embedding_size, hidden_size_list: list, mid_hidden):
        super(VAE, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size_list = hidden_size_list
        self.mid_hidden = mid_hidden

        self.enc_feature_size_list = [self.embedding_size] + self.hidden_size_list + [self.mid_hidden * 2]
        self.dec_feature_size_list = [self.embedding_size] + self.hidden_size_list + [self.mid_hidden]

        self.encoder = nn.ModuleList(
            [nn.Linear(self.enc_feature_size_list[i], self.enc_feature_size_list[i + 1]) for i in
             range(len(self.enc_feature_size_list) - 1)])
        self.decoder = nn.ModuleList(
            [nn.Linear(self.dec_feature_size_list[i], self.dec_feature_size_list[i - 1]) for i in
             range(len(self.dec_feature_size_list) - 1, 0, -1)])

    def encode(self, x):
        for i, layer in enumerate(self.encoder):
            x = self.encoder[i](x)
            if i != len(self.encoder) - 1:
                x = F.relu(x)
        return x

    def decode(self, x):
        for i, layer in enumerate(self.decoder):
            x = self.decoder[i](x)
            if i != len(self.decoder) - 1:
                x = F.relu(x)
        return x

    def forward(self, x):
        encoder_output = self.encode(x)
        mu, sigma = torch.chunk(encoder_output, 2, dim=1)  # mu, log_var
        hidden = torch.randn_like(sigma) + mu * torch.exp(sigma) ** 0.5  # var => std
        x_hat = self.decode(hidden)
        kl_div = 0.5 * torch.sum(torch.exp(sigma) + torch.pow(mu, 2) - 1 - sigma) / (x.shape[0] * x.shape[1])
        return x_hat, kl_div

    def get_hidden(self, x):
        encoder_output = self.encode(x)
        mu, sigma = torch.chunk(encoder_output, 2, dim=1)  # mu, log_var
        hidden = torch.randn_like(sigma) + mu * torch.exp(sigma) ** 0.5  # var => std
        hidden = hidden.detach()
        return hidden
