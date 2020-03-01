from models import VAE
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from datasets.load_features import load_cell_gene_features
from pathlib import Path


def main(params):
    feature_size = params.feature_size
    epoch_num = params.epoch_num
    lr = params.lr
    weight_decay = params.weight_decay
    hidden_list = params.hidden_list
    save_name = params.name
    device = torch.device('cuda:1')
    features = load_cell_gene_features(params)
    features = torch.tensor(features, dtype=torch.float32).to(device)
    vae = VAE(embedding_size=features.shape[1], hidden_size_list=hidden_list, mid_hidden=feature_size).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    for i in range(epoch_num):
        x_hat, kl_div = vae(features)
        loss = criterion(x_hat, features)

        if kl_div is not None:
            loss += kl_div

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"epoch {i}: {loss:.4f} loss.")

    proj_path = Path(__file__).parent.resolve().parent.resolve()
    model_path = proj_path / 'saved_model'
    if not model_path.exists():
        model_path.mkdir()
    torch.save(vae, model_path / save_name)


if __name__ == '__main__':
    """
    python ./code/train_vae.py --hidden_list 600 --train 3510 1311 6633 6905 4909 2081 --tissue Mammary_gland
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=10086)
    parser.add_argument('--feature_size', type=int, default=400)
    parser.add_argument('--pca_dim', type=int, default=1000)
    parser.add_argument('--epoch_num', type=int, default=18000)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--hidden_list', type=int, nargs='+', required=True)
    parser.add_argument('--tissue', required=True, type=str)
    parser.add_argument('--train', required=True, nargs='+')
    parser.add_argument('--name', type=str, default='vae.pkl')
    params = parser.parse_args()

    main(params)
