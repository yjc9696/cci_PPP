import argparse
import random
from pathlib import Path
from time import time

import dgl
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, vstack
from sklearn.decomposition import PCA


def load_PPI_mammary_gland(params):
    random_seed = params.random_seed
    dense_dim = params.dense_dim
    # root = params.root
    train = params.dataset
    train_dataset = params.train_dataset
    test_dataset = params.test_dataset
    each_dataset_size = params.each_dataset_size

    tissue = params.tissue

    ligand_receptor_pair_path = params.ligand_receptor_pair_path

    all_data = train

    proj_path = Path(__file__).parent.resolve().parent.resolve().parent.resolve()
    PPI_data_path = proj_path / 'data' / 'PPI'/tissue

    # data_path = PPI_data_path / f'{tissue}_GO_{train[0]}.lab'
    # print(data_path)
    # # load data file then update graph
    # df = pd.read_csv(data_path, sep='\t',skiprows=1,header=None)  # (gene, cell)

    edge_list_path = PPI_data_path / (tissue + '.edgelist')
    df = pd.read_csv(edge_list_path, sep="\t", header=None)
    df.head();
    edge_list = [];
    for indexs in df.index:
        rowData = df.loc[indexs].values[0:2]
        rowData = rowData.tolist()
        edge_list.append(rowData)
    print(len(edge_list));
    nodes = [];
    for edge in edge_list:
        if edge[0] not in nodes:
            nodes.append(edge[0])
        if edge[1] not in nodes:
            nodes.append(edge[1])
    nodes.sort()

    gene2id = {gene: idx for idx, gene in enumerate(nodes)}
    num_genes = len(nodes)
    train_cci_labels = []
    test_cci_labels = []

    # 1. read data, restore everything in a graph,
    graph = dgl.DGLGraph()
    # debug
    # graph.to(torch.device('cuda:0'))
    start = time()
    # add all genes as nodes

    graph.add_nodes(num_genes)
    # construct labels: -1 gene 0~19 cell types

    # add gene edges: ligand and receptor
    for i,j in edge_list:
        graph.add_edge(gene2id[i], gene2id[j])
        graph.add_edge(gene2id[j], gene2id[i])

    labels = {}
    matrices = []
    for num in all_data:
        # data_path = f'{root}/mouse_{tissue}{num}_data.csv'
        data_path = PPI_data_path / f'{tissue}_GO_{num}.lab'
        # load data file then update graph 
        df = pd.read_csv(data_path, sep='\t',skiprows=1,header=None) # (gene, cell)
        for indexs in df.index:
            rowData = df.loc[indexs].values[0:2]
            rowData = rowData.tolist()
            if(rowData[0] in nodes):
                labels[rowData[0]]=rowData[1]
                train_cci_labels+=rowData


        # add more nodes because of new cells
        # print(f'Added {len(df)} nodes and {len(src_idx)} edges.')
        print(f'#Nodes: {graph.number_of_nodes()}, #Edges: {graph.number_of_edges()}.')
        print(f'Costs {time() - start:.3f} s in total.\n')

    # 2. create features
    sparse_feat = vstack(matrices).toarray()  # cell-wise  (cell, gene)
    # transpose to gene-wise
    # sparse_feat = preprocessing.scale(sparse_feat, axis=1) #very good
    # sparse_feat = preprocessing.normalize(sparse_feat, norm='max', axis=1) 
    # sparse_feat = sparse_feat / np.linalg.norm(sparse_feat, axis=1)[0]
    # import pdb; pdb.set_trace()
    # sparse_feat = sparse_feat[:, 0:10000]
    print(sparse_feat.shape)
    gene_pca = PCA(dense_dim, random_state=random_seed).fit(sparse_feat[:sum(train)].T)
    gene_feat = gene_pca.transform(sparse_feat[:sum(train)].T)
    gene_evr = sum(gene_pca.explained_variance_ratio_) * 100
    # print(f'[PCA] explained_variance_: {gene_pca.explained_variance_}')
    print(f'[PCA] Gene EVR: {gene_evr:.2f} %.')
    # do normalization
    sparse_feat = sparse_feat / np.sum(sparse_feat, axis=1, keepdims=True)
    # sparse_feat = preprocessing.scale(sparse_feat, axis=1) #very good
    # sparse_feat = preprocessing.normalize(sparse_feat, norm='max', axis=1) 
    # sparse_feat = sparse_feat / np.linalg.norm(sparse_feat, axis=1)[0]

    # use weighted gene_feat as cell_feat
    cell_feat = sparse_feat.dot(gene_feat)
    gene_feat = torch.from_numpy(gene_feat)  # use shared storage
    cell_feat = torch.from_numpy(cell_feat)

    features = torch.cat([gene_feat, cell_feat], dim=0).type(torch.float)
    # features = torch.FloatTensor(graph.number_of_nodes(), params.dense_dim).normal_()

    # 3. then create masks for different purposes.

    num_cells = graph.number_of_nodes() - num_genes

    train_cci_labels = torch.LongTensor(train_cci_labels)
    test_cci_labels = torch.LongTensor(test_cci_labels)

    num_pairs = len(train_cci_labels)
    print(f"Total train {len(train_cci_labels)} pairs.")
    print(f"Total test {len(test_cci_labels)} pairs.")

    train_mask = torch.zeros(num_pairs, dtype=torch.int32)
    vali_mask = torch.zeros(num_pairs, dtype=torch.int32)

    # import pdb;pdb.set_trace()
    split_mask = random.sample(range(0, num_pairs), int(0.8 * num_pairs))
    train_mask[split_mask] += 1
    vali_mask = torch.where(train_mask > 0, torch.full_like(train_mask, 0), torch.full_like(train_mask, 1))

    assert train_mask.sum().item() + vali_mask.sum().item() == num_pairs
    train_mask = train_mask.type(torch.bool)
    vali_mask = vali_mask.type(torch.bool)

    # return num_cells, num_genes, num_labels, graph, features, train_cci_labels, train_mask, vali_mask
    return num_cells, num_genes, 2, graph, features, train_cci_labels, train_mask, vali_mask, test_cci_labels


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
    parser.add_argument("--dataset", nargs="+", required=True, type=int,
                        help="list of dataset id")
    # parser.add_argument("--test_dataset", nargs="+", required=True, type=int,
    #                     help="list of dataset id")
    parser.add_argument("--tissue", required=True, type=str,
                        help="list of dataset id")

    params = parser.parse_args()

    load_PPI_mammary_gland(params)
