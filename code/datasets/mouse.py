import argparse

import pandas as pd
import dgl
from time import time
import torch
from scipy.sparse import csr_matrix, vstack
from sklearn.decomposition import PCA
from pathlib import Path
import numpy as np
import random


def load_mouse_mammary_gland(params):
    random_seed = params.random_seed
    dense_dim = params.dense_dim
    # root = params.root
    train = params.train_dataset
    test = params.test_dataset
    tissue = params.tissue

    all_data = train + test

    proj_path = Path(__file__).parent.resolve().parent.resolve().parent.resolve()
    mouse_data_path = proj_path / 'data' / 'mouse_data'
    statistics_path = mouse_data_path / 'statistics'

    if not statistics_path.exists():
        statistics_path.mkdir()

    gene_statistics_path = statistics_path / (tissue + '_genes.txt')
    cell_statistics_path = statistics_path / (tissue + '_cell_type.txt')

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

    # generate cell label statistics file
    if not cell_statistics_path.exists():
        cell_files = mouse_data_path.glob(f'*{tissue}*_celltype.csv')
        cell_types = set()
        for file in cell_files:
            # import pdb; pdb.set_trace()
            cell_types = set(pd.read_csv(file, dtype=np.str, header=0).values[:, 2]) | cell_types
        id2label = list(cell_types)
        with open(cell_statistics_path, 'w', encoding='utf-8') as f:
            for cell_type in id2label:
                f.write(cell_type + '\r\n')
    else:
        id2label = []
        with open(cell_statistics_path, 'r', encoding='utf-8') as f:
            for line in f:
                id2label.append(line.strip())

    # prepare unified genes
    gene2id = {gene: idx for idx, gene in enumerate(id2gene)}
    num_genes = len(id2gene)

    # prepare unified labels
    num_labels = len(id2label)
    label2id = {label: idx for idx, label in enumerate(id2label)}
    print(f"totally {num_genes} genes, {num_labels} labels.")

    # 1. read data, restore everything in a graph,
    graph = dgl.DGLGraph()
    #debug
    # graph.to(torch.device('cuda:0'))
    start = time()
    # add all genes as nodes
    graph.add_nodes(num_genes)
    # construct labels: -1 gene 0~19 cell types
    labels = []
    matrices = []
    for num in all_data:
        # data_path = f'{root}/mouse_{tissue}{num}_data.csv'
        data_path = mouse_data_path / f'mouse_{tissue}{num}_data.csv'
        type_path = mouse_data_path / f'mouse_{tissue}{num}_celltype.csv'

        # load celltype file then update labels accordingly
        cell2type = pd.read_csv(type_path, index_col=0)
        cell2type.columns = ['cell', 'type']
        cell2type['id'] = cell2type['type'].map(label2id)
        assert not cell2type['id'].isnull().any(), 'something wrong about celltype file.'
        labels += cell2type['id'].tolist()

        # load data file then update graph 
        df = pd.read_csv(data_path, index_col=0)  # (gene, cell)
        df = df.transpose(copy=True)  # (cell, gene)
        assert cell2type['cell'].tolist() == df.index.tolist()
        df = df.rename(columns=gene2id)
        # filter out useless columns if exists (when using gene intersection)
        col = [c for c in df.columns if c in gene2id.values()]
        df = df[col]
        # print(df.head())
        print(f'Nonzero Ratio: {df.fillna(0).astype(bool).sum().sum() / df.size * 100:.2f}%')

        # import pdb; pdb.set_trace()
        # maintain inter-datasets index for graph and RNA-seq values
        arr = df.to_numpy()
        row_idx, col_idx = arr.nonzero()  # intra-dataset index
        non_zeros = arr[(row_idx, col_idx)]  # non-zero values
        # inter-dataset index
        src_idx = row_idx + graph.number_of_nodes()  # cell_index
        tgt_idx = df.columns[col_idx].astype(int).tolist()  # gene_index
        info_shape = (len(df), num_genes)
        info = csr_matrix((non_zeros, (row_idx, tgt_idx)), shape=info_shape)
        matrices.append(info)

        # add more nodes because of new cells
        graph.add_nodes(len(df))
        graph.add_edges(src_idx, tgt_idx)
        graph.add_edges(tgt_idx, src_idx)
        print(f'Added {len(df)} nodes and {len(src_idx)} edges.')
        print(f'#Nodes: {graph.number_of_nodes()}, #Edges: {graph.number_of_edges()}.')
        print(f'Costs {time() - start:.3f} s in total.\n')

    # 2. create features
    sparse_feat = vstack(matrices).toarray()  # cell-wise  (cell, gene)
    # transpose to gene-wise
    gene_pca = PCA(dense_dim, random_state=random_seed).fit(sparse_feat[:sum(train)].T)
    gene_feat = gene_pca.transform(sparse_feat[:sum(train)].T)
    gene_evr = sum(gene_pca.explained_variance_ratio_) * 100
    print(f'[PCA] Gene EVR: {gene_evr:.2f} %.')
    # do normalization
    sparse_feat = sparse_feat / np.sum(sparse_feat, axis=1, keepdims=True)
    # use weighted gene_feat as cell_feat
    cell_feat = sparse_feat.dot(gene_feat)
    gene_feat = torch.from_numpy(gene_feat)  # use shared storage
    cell_feat = torch.from_numpy(cell_feat)

    features = torch.cat([gene_feat, cell_feat], dim=0).type(torch.float)
    # features = torch.FloatTensor(graph.number_of_nodes(), params.dense_dim).normal_()
    # 3. then create masks for different purposes.
    labels = torch.LongTensor(labels)

    num_cells = graph.number_of_nodes() - num_genes
    train_mask = torch.zeros(num_cells + num_genes, dtype=torch.int32)
    test_mask = torch.zeros(num_cells + num_genes, dtype=torch.int32)

    # import pdb;pdb.set_trace()
    split_mask = random.sample(range(num_genes, num_genes+num_cells), sum(train))
    train_mask[split_mask] += 1
    test_mask = torch.where(train_mask>0, torch.full_like(train_mask, 0), torch.full_like(train_mask, 1))
    test_mask[[i for i in range(num_genes)]] = 0


    # train_mask[num_genes:sum(train) + num_genes] += 1
    # test_mask[-sum(test):] += 1
    # train_nid = torch.where(train_mask == 1)[0]
    # test_nid = torch.where(test_mask == 1)[0]
    assert train_mask.sum().item() + test_mask.sum().item() == num_cells
    train_mask = train_mask.type(torch.bool)
    test_mask = test_mask.type(torch.bool)
    return num_cells, num_genes, num_labels, graph, features, labels, train_mask, test_mask


if __name__ == '__main__':
    """
    python ./code/datasets/mouse.py --train_dataset 3510 --test_dataset 1059 --tissue Mammary_gland
    python ./code/datasets/mouse.py --train_dataset 3510 1311 6633 6905 4909 2081 --test_dataset 1059 648 1592 --tissue Mammary_gland
    """
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
    parser.add_argument("--root", type=str, default="../../data/mammary_gland",
                        help="root path")
    parser.add_argument("--train_dataset", nargs="+", required=True, type=int,
                        help="list of dataset id")
    parser.add_argument("--test_dataset", nargs="+", required=True, type=int,
                        help="list of dataset id")
    parser.add_argument("--tissue", required=True, type=str,
                        help="list of dataset id")

    params = parser.parse_args()

    load_mouse_mammary_gland(params)
