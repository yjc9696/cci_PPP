# newer than mouse.py
# this file use the analyze.csv 
# format: id1, id2, type1, type2, cci, mask_num, score, max_score, cell_name1, cell_name2

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
from sklearn import preprocessing
import re
from .dataset import TrainSet



def load_mouse_mammary_gland_bone_marrow(params):
    print('using mouse2_bone_marrow.py')
    random_seed = params.random_seed
    dense_dim = params.dense_dim
    score_limit = params.score_limit

    proj_path = Path(__file__).parent.resolve().parent.resolve().parent.resolve()
    mouse_data_path = proj_path / 'data' / params.data_dir #small_intestine
    train_dataset = mouse_data_path / params.train_dataset
    test_dataset = mouse_data_path / params.test_dataset

    ligand_receptor_path = train_dataset / params.ligand_receptor_gene

    # prepare the ligand and receptor genes.
    all_dataset = [train_dataset, test_dataset]
    genes = set()
    for dataset in all_dataset:
        data_path = dataset / params.cell_data_path
        # cell * gene
        cell_data = pd.read_csv(data_path, index_col=0).transpose().fillna(0)
        for ge in cell_data.columns.tolist():
            genes.add(ge)

    # prepare ligand receptor pair
    lcp = pd.read_csv(ligand_receptor_path, header=0, index_col=0)
    # lcp = lcp.applymap(lambda x: gene2id[x] if x in gene2id else -1)
    ligand = lcp['ligand'].tolist()
    receptor = lcp['receptor'].tolist()
    lcp_genes = set(lcp['ligand']) | set(lcp['receptor'])
    gene_inter = lcp_genes.intersection(genes)

    exist_ligand = list()
    exist_receptor = list()
    for i in range(len(ligand)):
        if (ligand[i] in genes and receptor[i] in genes) and (ligand[i] not in exist_ligand and receptor[i] not in exist_receptor):
            exist_ligand.append(ligand[i])
            exist_receptor.append(receptor[i])
    pair1_mask = exist_ligand + exist_receptor
    pair2_mask = exist_receptor + exist_ligand

    assert(len(exist_ligand) == len(exist_receptor), "ligand num should match receptor num.")

    # attention: only use the ligand and receptor genes. the other genes are ignored.
    gene2id = {gene:idx for idx, gene in enumerate(gene_inter)}
    num_genes = len(gene2id)

    print(f"totally {num_genes} ligand and receptor genes. {len(exist_ligand)} pairs.")

    # import pdb; pdb.set_trace()

    # 1. read data, restore everything in a graph,
    graph = dgl.DGLGraph()
    graph_test = dgl.DGLGraph()

    start = time()
    # add all genes as nodes
    graph.add_nodes(num_genes)
    graph_test.add_nodes(num_genes)
    # construct labels: -1 gene 0~19 cell types

    # add gene edges: ligand and receptor
    for i, j in zip(exist_ligand, exist_receptor):
        graph.add_edge(gene2id[i], gene2id[j])
        graph.add_edge(gene2id[j], gene2id[i])
        graph_test.add_edge(gene2id[i], gene2id[j])
        graph_test.add_edge(gene2id[j], gene2id[i])

    matrices = []
    matrices_test = []
    is_test_dataset = 0
    train_cci_labels = []
    test_cci_labels = []

    for dataset in all_dataset:

        data_path = dataset.glob('*data.csv').__next__()

        cci_labels = []
        # load data file then update graph 
        df = pd.read_csv(data_path, index_col=0)  # (gene, cell)
        df = df.fillna(0)

        df = df.transpose(copy=True)  # (cell, gene)

        df = df.rename(columns=gene2id)
        # filter out useless columns if exists (when using gene intersection)
        col = [c for c in df.columns if c in gene2id.values()]
        df = df[col]

        print(f'Nonzero Ratio: {df.fillna(0).astype(bool).sum().sum() / df.size * 100:.2f}%')

        # import pdb; pdb.set_trace()

        # stores the pairs that have relation
        # indexs = list()
        cci_labels_paths = dataset.glob('*analyze*.csv').__next__()
        cci_labels_df = pd.read_csv(cci_labels_paths).groupby('cci')

        gt_cci_labels = cci_labels_df.get_group(1)
        gt_cci_labels = gt_cci_labels[gt_cci_labels['score'] > score_limit]
        # 3, 4记录cell真实id，方便构建cell cell interaction
        gt_cci_labels_data = pd.DataFrame(columns=[0,1,2,3,4,5])
        gt_cci_labels_data[0] = gt_cci_labels['id1']
        gt_cci_labels_data[1] = gt_cci_labels['id2']
        # import pdb; pdb.set_trace()
        gt_cci_labels_data[2] = 1
        gt_cci_labels_data[3] = gt_cci_labels['score']
        gt_cci_labels_data[4] = gt_cci_labels_data[0] # truth id1
        gt_cci_labels_data[5] = gt_cci_labels_data[1] # truth id2
        gt_cci_labels_data[0] = gt_cci_labels_data[0].apply(lambda x: x+graph.number_of_nodes())
        gt_cci_labels_data[1] = gt_cci_labels_data[1].apply(lambda x: x+graph.number_of_nodes())

        cur_cci_labels = gt_cci_labels_data[[0,1,2,3,4,5]].values.tolist()
        cur_cci_labels = np.asarray(cur_cci_labels)
        cur_cci_labels = cur_cci_labels.tolist()
        cci_labels += cur_cci_labels

        # import pdb; pdb.set_trace()
        try:
            junk_cci_labels = cci_labels_df.get_group(0)
            junk_cci_labels = junk_cci_labels[junk_cci_labels['score'] > score_limit]
            # 3, 4记录cell真实id，方便构建cell cell interaction
            junk_cci_labels_data = pd.DataFrame(columns=[0,1,2,3,4,5])
            junk_cci_labels_data[0] = junk_cci_labels['id1']
            junk_cci_labels_data[1] = junk_cci_labels['id2']
            junk_cci_labels_data[2] = 0
            junk_cci_labels_data[3] = junk_cci_labels['score'] * (-1)
            junk_cci_labels_data[4] = junk_cci_labels_data[0]
            junk_cci_labels_data[5] = junk_cci_labels_data[1]
            junk_cci_labels_data[0] = junk_cci_labels_data[0].apply(lambda x: x+graph.number_of_nodes())
            junk_cci_labels_data[1] = junk_cci_labels_data[1].apply(lambda x: x+graph.number_of_nodes())

            cur_cci_labels = junk_cci_labels_data[[0,1,2,3,4,5]].values.tolist()
            cur_cci_labels = np.asarray(cur_cci_labels)
            cur_cci_labels = cur_cci_labels.tolist()
            cci_labels += cur_cci_labels
        except Exception as e:
            pass
        
        junk_cci_labels = cci_labels_df.get_group(-1)
        junk_cci_labels = junk_cci_labels[junk_cci_labels['score'] > score_limit]
        # 3, 4记录cell真实id，方便构建cell cell interaction
        junk_cci_labels_data = pd.DataFrame(columns=[0,1,2,3,4,5])
        junk_cci_labels_data[0] = junk_cci_labels['id1']
        junk_cci_labels_data[1] = junk_cci_labels['id2']
        junk_cci_labels_data[2] = 0
        junk_cci_labels_data[3] = junk_cci_labels['score'] * (-1)
        junk_cci_labels_data[4] = junk_cci_labels_data[0]
        junk_cci_labels_data[5] = junk_cci_labels_data[1]
        junk_cci_labels_data[0] = junk_cci_labels_data[0].apply(lambda x: x+graph.number_of_nodes())
        junk_cci_labels_data[1] = junk_cci_labels_data[1].apply(lambda x: x+graph.number_of_nodes())

        cur_cci_labels = junk_cci_labels_data[[0,1,2,3,4,5]].values.tolist()
        cur_cci_labels = np.asarray(cur_cci_labels)
        cur_cci_labels = cur_cci_labels.tolist()
        cci_labels += cur_cci_labels
        print(f'unkown cci labels: {len(cur_cci_labels)}')

        # maintain inter-datasets index for graph and RNA-seq values
        arr = df.to_numpy()
        row_idx, col_idx = arr.nonzero()  # intra-dataset index
        non_zeros = arr[(row_idx, col_idx)]  # non-zero values
        # inter-dataset index
        src_idx = row_idx + graph.number_of_nodes()  # cell_index
        tgt_idx = df.columns[col_idx].astype(int).tolist()  # gene_index
        info_shape = (len(df), num_genes)
        info = csr_matrix((non_zeros, (row_idx, tgt_idx)), shape=info_shape)
        # import pdb; pdb.set_trace()

        if not is_test_dataset:
            train_cci_labels += cci_labels
            matrices.append(info)
            # add more nodes because of new cells
            graph.add_nodes(len(df))
            graph.add_edges(src_idx, tgt_idx)
            graph.add_edges(tgt_idx, src_idx)
        else:
            test_cci_labels += cci_labels

        matrices_test.append(info)
        graph_test.add_nodes(len(df))
        graph_test.add_edges(src_idx, tgt_idx)
        graph_test.add_edges(tgt_idx, src_idx)

        print(f'Added {len(df)} nodes and {len(src_idx)} edges.')
        print(f'#Nodes: {graph_test.number_of_nodes()}, #Edges: {graph_test.number_of_edges()}.')
        print(f'Costs {time() - start:.3f} s in total.\n')

        is_test_dataset = 1

    # 3. create test features
    sparse_feat_test = vstack(matrices_test).toarray()  # cell-wise  (cell, gene)
    # transpose to gene-wise
    print(sparse_feat_test.shape)
    gene_pca = PCA(dense_dim, random_state=random_seed).fit(sparse_feat_test.T)
    gene_feat = gene_pca.transform(sparse_feat_test.T)
    gene_evr = sum(gene_pca.explained_variance_ratio_) * 100
    # print(f'[PCA] explained_variance_: {gene_pca.explained_variance_}')
    print(f'[PCA] Gene EVR: {gene_evr:.2f} %.')
    # do normalization
    sparse_feat_test = sparse_feat_test / np.sum(sparse_feat_test, axis=1, keepdims=True)

    # use weighted gene_feat as cell_feat
    cell_feat = sparse_feat_test.dot(gene_feat)
    gene_feat_tensor = torch.from_numpy(gene_feat)  # use shared storage
    cell_feat = torch.from_numpy(cell_feat)

    features_test = torch.cat([gene_feat_tensor, cell_feat], dim=0).type(torch.float)

# 2. create features
    sparse_feat = vstack(matrices).toarray()  # cell-wise  (cell, gene)
    # use weighted gene_feat as cell_feat
    cell_feat = sparse_feat.dot(gene_feat)
    gene_feat = torch.from_numpy(gene_feat)  # use shared storage
    cell_feat = torch.from_numpy(cell_feat)

    features = torch.cat([gene_feat, cell_feat], dim=0).type(torch.float)


    # 4. then create masks for different purposes.
    num_cells = graph.number_of_nodes() - num_genes
    # import pdb; pdb.set_trace()
    train_cci_labels = torch.LongTensor(train_cci_labels)
    test_cci_labels = torch.LongTensor(test_cci_labels)

    num_pairs = len(train_cci_labels)
    print(f"Total train {len(train_cci_labels)} pairs.")
    print(f"Total test {len(test_cci_labels)} pairs.")
    print(f'train positive ratio: {train_cci_labels[:,2].sum().item() / len(train_cci_labels)}')
    print(f'test positive ratio: {test_cci_labels[:,2].sum().item() / len(test_cci_labels)}')
    # import pdb; pdb.set_trace()
    train_mask = torch.zeros(num_pairs, dtype=torch.int32)
    vali_mask = torch.zeros(num_pairs, dtype=torch.int32)

    split_mask = random.sample(range(0, num_pairs), int(0.8*num_pairs))
    train_mask[split_mask] += 1
    vali_mask = torch.where(train_mask > 0, torch.full_like(train_mask, 0), torch.full_like(train_mask, 1))

    assert train_mask.sum().item() + vali_mask.sum().item() == num_pairs
    train_mask = train_mask.type(torch.bool)
    vali_mask = vali_mask.type(torch.bool)

    # import pdb; pdb.set_trace()

    # return num_cells, num_genes, num_labels, graph, features, train_cci_labels, train_mask, vali_mask
    return num_cells, num_genes, 2, graph, features, graph_test, features_test, \
        train_cci_labels, train_mask, vali_mask, test_cci_labels


if __name__ == '__main__':
    """
    python ./code/datasets/mouse.py --train_dataset 3510 --test_dataset 1059 --tissue Mammary_gland
    python ./code/datasets/mouse.py --train_dataset 3510 1311 6633 6905 4909 2081 --test_dataset 1059 648 1592 --tissue Mammary_gland
    """
    parser = argparse.ArgumentParser(description='GraphSAGE')
    parser.add_argument("--random_seed", type=int, default=10086)
    parser.add_argument("--dense_dim", type=int, default=400,
                        help="number of hidden gcn units")
    parser.add_argument("--hidden_dim", type=int, default=200,
                        help="number of hidden gcn units")
    parser.add_argument("--ligand_receptor_pair_path", type=str, default="mouse_ligand_receptor_pair",
                        help="gene ligand receptor pair path")
    parser.add_argument("--train_dataset", type=str, default="train_dataset",
                        help="train dataset")
    parser.add_argument("--test_dataset", type=str, default="test_dataset",
                        help="test dataset")
    parser.add_argument("--each_dataset_size", type=int, default=0,
                        help="0 represent all")
    parser.add_argument("--ligand_receptor_gene", type=str, default='mouse_ligand_receptor_pair.csv',
                        help="cluster - cluster interaction depleted")
    parser.add_argument("--data_dir", type=str, default='mouse_small_intestine',
                        help="root path of the data dir")
    parser.add_argument("--cell_data_path", type=str, default='mouse_small_intestine_1189_data.csv',
                        help="cell data gene")
    params = parser.parse_args()
    print(params)

    load_mouse_mammary_gland_bone_marrow(params)
