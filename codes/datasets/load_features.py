import argparse

import pandas as pd
import torch
from scipy.sparse import csr_matrix, vstack
from sklearn.decomposition import PCA
from pathlib import Path
import numpy as np


def load_cell_gene_features(params):
    """
    return: (cell, pca_dim)
    """
    random_seed = params.random_seed
    pca_dim = params.pca_dim
    train = params.train
    tissue = params.tissue

    proj_path = Path(__file__).parent.resolve().parent.resolve().parent.resolve()
    mouse_data_path = proj_path / 'data' / 'mouse_data'
    statistics_path = mouse_data_path / 'statistics'

    if not statistics_path.exists():
        statistics_path.mkdir()

    gene_statistics_path = statistics_path / (tissue + '_genes.txt')

    # generate gene statistics file
    if not gene_statistics_path.exists():
        data_files = mouse_data_path.glob(f'*{tissue}*_data.csv')
        genes = None
        for file in data_files:
            data = pd.read_csv(file, dtype=np.str, header=0).values[:, 0]
            if genes is None:
                genes = set(data)
            else:
                genes = genes | set(data)
        id2gene = list(genes)
        id2gene.sort()
        with open(gene_statistics_path, 'w', encoding='utf-8') as f:
            for gene in id2gene:
                f.write(gene + '\r\n')
    else:
        id2gene = []
        with open(gene_statistics_path, 'r', encoding='utf-8') as f:
            for line in f:
                id2gene.append(line.strip())

    # prepare unified genes
    gene2id = {gene: idx for idx, gene in enumerate(id2gene)}
    num_genes = len(id2gene)
    print(f"totally {num_genes} genes.")

    matrices = []
    for num in train:
        data_path = mouse_data_path / f'mouse_{tissue}{num}_data.csv'

        # load data file then update graph
        df = pd.read_csv(data_path, index_col=0)  # (gene, cell)
        df = df.transpose(copy=True)  # (cell, gene)
        df = df.rename(columns=gene2id)
        # filter out useless columns if exists (when using gene intersection)
        col = [c for c in df.columns if c in gene2id.values()]
        df = df[col]

        # maintain inter-datasets index for graph and RNA-seq values
        arr = df.to_numpy()
        print(arr.shape)
        row_idx, col_idx = arr.nonzero()  # intra-dataset index
        non_zeros = arr[(row_idx, col_idx)]  # non-zero values
        # inter-dataset index
        tgt_idx = df.columns[col_idx].astype(int).tolist()  # gene_index
        info_shape = (len(df), num_genes)
        info = csr_matrix((non_zeros, (row_idx, tgt_idx)), shape=info_shape)
        matrices.append(info)
    sparse_feat = vstack(matrices).toarray()  # cell-wise  (cell, gene)
    gene_pca = PCA(pca_dim, random_state=random_seed).fit(sparse_feat.T)
    gene_feat = gene_pca.transform(sparse_feat.T)  # (cell, pca_dim)
    gene_evr = sum(gene_pca.explained_variance_ratio_) * 100
    print(f'[PCA] Gene EVR: {gene_evr:.2f} %.')
    return gene_feat


if __name__ == '__main__':
    """
    python ./code/datasets/load_features.py --train 3510 1311 6633 6905 4909 2081 --tissue Mammary_gland
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--tissue', required=True, type=str)
    parser.add_argument('--train', required=True, nargs='+')
    params = parser.parse_args()

    load_cell_gene_features(params)
