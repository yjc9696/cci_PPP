import argparse
import random
from pathlib import Path
from time import time

import dgl
import pandas as pd
import torch
from scipy.sparse import coo_matrix, vstack
from sklearn.decomposition import PCA


def load_PPP_mammary_gland(params):
    random_seed = params.random_seed
    dense_dim = params.dense_dim
    # root = params.root
    train = params.dataset[0]
    train_dataset = params.train_dataset
    test_dataset = params.test_dataset
    each_dataset_size = params.each_dataset_size

    tissue = params.tissue

    ligand_receptor_pair_path = params.ligand_receptor_pair_path


    proj_path = Path(__file__).parent.resolve().parent.resolve().parent.resolve()
    data_path = proj_path / 'data' / 'PPP'

    # data_path = PPI_data_path / f'{tissue}_GO_{train[0]}.lab'
    # print(data_path)
    # # load data file then update graph
    # df = pd.read_csv(data_path, sep='\t',skiprows=1,header=None)  # (gene, cell)

    edge_list_path = data_path / ('PP-'+str(train)+'.csv')
    df = pd.read_csv(edge_list_path, header=None)
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

    # stores the pairs that have relation

    indexs = list()
    train_cci_labels_gt_paths = (data_path / train_dataset).glob('*gt*.csv')
    for file in sorted(train_cci_labels_gt_paths):
        cur_train_cci_labels = pd.read_csv(file, header=None)
        cur_train_cci_labels[0] = cur_train_cci_labels[0].apply(lambda x:gene2id[x])
        cur_train_cci_labels[1] = cur_train_cci_labels[1].apply(lambda x:gene2id[x])
        cur_train_cci_labels[2] = 1
        cur_train_cci_labels = cur_train_cci_labels.values.tolist()
        train_cci_labels += cur_train_cci_labels
    for i,j,k in train_cci_labels:
        graph.add_edge(i, j)
        graph.add_edge(i, j)
    junk_labels_path = (data_path / train_dataset).glob('*junk*.csv')
    for file in sorted(junk_labels_path):
        junk_cci_labels = pd.read_csv(file, header=None)
        junk_cci_labels[0] = junk_cci_labels[0].apply(lambda x:gene2id[x])
        junk_cci_labels[1] = junk_cci_labels[1].apply(lambda x:gene2id[x])
        junk_cci_labels[2] = 0
        junk_cci_labels = junk_cci_labels.values.tolist()
        train_cci_labels += junk_cci_labels

    test_cci_labels_gt_paths = (data_path / test_dataset).glob('*gt*.csv')
    for file in sorted(test_cci_labels_gt_paths):
        cur_test_cci_labels = pd.read_csv(file, header=None)
        cur_test_cci_labels[0] = cur_test_cci_labels[0].apply(lambda x:gene2id[x])
        cur_test_cci_labels[1] = cur_test_cci_labels[1].apply(lambda x:gene2id[x])
        cur_test_cci_labels[2] = 1
        cur_test_cci_labels = cur_test_cci_labels.values.tolist()
        test_cci_labels += cur_test_cci_labels
    junk_labels_path = (data_path / test_dataset).glob('*junk*.csv')
    for file in sorted(junk_labels_path):
        junk_cci_labels = pd.read_csv(file, header=None)
        junk_cci_labels[0] = junk_cci_labels[0].apply(lambda x:gene2id[x])
        junk_cci_labels[1] = junk_cci_labels[1].apply(lambda x:gene2id[x])
        junk_cci_labels[2] = 0
        junk_cci_labels = junk_cci_labels.values.tolist()
        test_cci_labels += junk_cci_labels
    random.shuffle(train_cci_labels)
    random.shuffle(test_cci_labels)
    matrices = []
    row_idx = []
    col_idx = []
    one = []  # intra-dataset index
    for edge in edge_list:
        row_idx.append(gene2id[edge[0]])
        col_idx.append(gene2id[edge[1]])
        one.append(1)
    # inter-dataset index
    info_shape = (num_genes, num_genes)
    info = coo_matrix((one, (row_idx, col_idx)), shape=info_shape)
    matrices.append(info)
    # 2. create features
    sparse_feat = vstack(matrices).toarray()  # cell-wise  (cell, gene)
    # transpose to gene-wise
    # sparse_feat = preprocessing.scale(sparse_feat, axis=1) #very good
    # sparse_feat = preprocessing.normalize(sparse_feat, norm='max', axis=1)
    # sparse_feat = sparse_feat / np.linalg.norm(sparse_feat, axis=1)[0]
    # import pdb; pdb.set_trace()
    # sparse_feat = sparse_feat[:, 0:10000]
    print(sparse_feat.shape)
    gene_pca = PCA(dense_dim, random_state=random_seed).fit(sparse_feat[:num_genes].T)
    # gene_pca = PCA(n_components='mle', random_state=random_seed).fit(sparse_feat[:sum(train)].T)

    gene_feat = gene_pca.transform(sparse_feat[:num_genes].T)
    gene_evr = sum(gene_pca.explained_variance_ratio_) * 100
    # print(f'[PCA] explained_variance_: {gene_pca.explained_variance_}')
    print(f'[PCA] Gene EVR: {gene_evr:.2f} %.')
    # do normalization
    # sparse_feat = sparse_feat / np.sum(sparse_feat, axis=1, keepdims=True)
    # sparse_feat = preprocessing.scale(sparse_feat, axis=1) #very good
    # sparse_feat = preprocessing.normalize(sparse_feat, norm='max', axis=1)
    # sparse_feat = sparse_feat / np.linalg.norm(sparse_feat, axis=1)[0]
    gene_feat = torch.from_numpy(gene_feat)
    features = torch.cat([gene_feat], dim=0).type(torch.float)
    # features = gene_feat.type(torch.float)
    # features = torch.FloatTensor(graph.number_of_nodes(), params.dense_dim).normal_()

    # 3. then create masks for different purposes.

    num_cells = num_genes

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

    load_PPP_mammary_gland(params)
