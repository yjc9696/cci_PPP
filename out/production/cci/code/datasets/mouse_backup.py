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

from .dataset import TrainSet



def load_mouse_mammary_gland(params):
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
    mouse_data_path = proj_path / 'data' / 'cell_cell_interaction'
    statistics_path = mouse_data_path / 'statistics'


    ligand_receptor_pair_path = mouse_data_path / (ligand_receptor_pair_path + '.csv')


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
        cell_files = mouse_data_path.glob(f'*{tissue}*_cellcluster.csv')
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

    # prepare ligand receptor pair
    lcp = pd.read_csv(ligand_receptor_pair_path, header=0, index_col=0)
    lcp = lcp.applymap(lambda x: gene2id[x] if x in gene2id else -1)
    ligand = lcp['ligand'].tolist()
    receptor = lcp['receptor'].tolist()
    

    genes = set(gene2id.keys())
    exist_ligand = list()
    exist_receptor = list()
    for i in range(len(ligand)):
        if ligand[i] in genes and receptor[i] in genes:
            exist_ligand.append(ligand[i])
            exist_receptor.append(receptor[i])
    pair1_mask = exist_ligand + exist_receptor
    pair2_mask = exist_receptor + exist_ligand
    # assert(len(ligand) == len(receptor), "ligand num should match receptor num.")


    train_cci_labels = []
    test_cci_labels = []

    # 1. read data, restore everything in a graph,
    graph = dgl.DGLGraph()
    #debug
    # graph.to(torch.device('cuda:0'))
    start = time()
    # add all genes as nodes
    graph.add_nodes(num_genes)
    # construct labels: -1 gene 0~19 cell types
    
    # add gene edges: ligand and receptor
    # for i, j in zip(exist_ligand, exist_receptor):
    #     graph.add_edge(gene2id[i], gene2id[j])
    #     graph.add_edge(gene2id[j], gene2id[i])

    labels = []
    matrices = []
    for num in all_data:
        # data_path = f'{root}/mouse_{tissue}{num}_data.csv'
        data_path = mouse_data_path / f'mouse_{tissue}_{num}_data.csv'
        type_path = mouse_data_path / f'mouse_{tissue}_{num}_cellcluster.csv'

        # load celltype file then update labels accordingly
        cell2type = pd.read_csv(type_path, index_col=0, dtype=str)
        cell2type.columns = ['cell', 'type']
        cell2type['id'] = cell2type['type'].map(label2id)

        assert not cell2type['id'].isnull().any(), 'something wrong about celltype file.'
        labels += cell2type['id'].tolist()

        # load data file then update graph 
        df = pd.read_csv(data_path, index_col=0)  # (gene, cell)
        df = df.fillna(0)
        df = df.transpose(copy=True)  # (cell, gene)
        assert cell2type['cell'].tolist() == df.index.tolist()
        df = df.rename(columns=gene2id)
        # filter out useless columns if exists (when using gene intersection)
        col = [c for c in df.columns if c in gene2id.values()]
        df = df[col]

        print(f'Nonzero Ratio: {df.fillna(0).astype(bool).sum().sum() / df.size * 100:.2f}%')


        # stores the pairs that have relation

        indexs = list()
        train_cci_labels_gt_paths = (mouse_data_path / train_dataset).glob('*gt*.csv')

        for file in sorted(train_cci_labels_gt_paths):
            cur_train_cci_labels = pd.read_csv(file, header=None)
            cur_train_cci_labels[0] = cur_train_cci_labels[0].apply(lambda x: x+graph.number_of_nodes())
            cur_train_cci_labels[1] = cur_train_cci_labels[1].apply(lambda x: x+graph.number_of_nodes())
            cur_train_cci_labels = cur_train_cci_labels.values.tolist()
     
            if each_dataset_size > 0 and len(cur_train_cci_labels) > each_dataset_size:
                cur_train_cci_labels = np.asarray(cur_train_cci_labels)
                index = np.random.choice(cur_train_cci_labels.shape[0]-1, each_dataset_size)
                indexs.append(index)
                cur_train_cci_labels = cur_train_cci_labels[index].tolist()

            train_cci_labels += cur_train_cci_labels

        junk_labels_path = (mouse_data_path / train_dataset).glob('*junk*.csv')
        
        cur_index = 0
        for file in sorted(junk_labels_path):
            junk_cci_labels = pd.read_csv(file, header=None)
            junk_cci_labels[0] = junk_cci_labels[0].apply(lambda x: x+graph.number_of_nodes())
            junk_cci_labels[1] = junk_cci_labels[1].apply(lambda x: x+graph.number_of_nodes())
            junk_cci_labels = junk_cci_labels.values.tolist()

            if each_dataset_size > 0 and len(junk_cci_labels) > each_dataset_size:
                junk_cci_labels = np.asarray(junk_cci_labels)
                # index = np.random.choice(junk_cci_labels.shape[0], len(junk_cci_labels)*0.5)
                # use the same index in gt
                junk_cci_labels = junk_cci_labels[indexs[cur_index]].tolist()
                cur_index += 1
            train_cci_labels += junk_cci_labels


        test_cci_labels_gt_paths = (mouse_data_path / test_dataset).glob('*gt*.csv')
        
        for file in test_cci_labels_gt_paths:
            cur_test_cci_labels = pd.read_csv(file, header=None)
            cur_test_cci_labels[0] = cur_test_cci_labels[0].apply(lambda x: x+graph.number_of_nodes())
            cur_test_cci_labels[1] = cur_test_cci_labels[1].apply(lambda x: x+graph.number_of_nodes())
            cur_test_cci_labels = cur_test_cci_labels.values.tolist()
            
            test_cci_labels += cur_test_cci_labels

        junk_labels_path = (mouse_data_path / test_dataset).glob('*junk*.csv')
        for file in junk_labels_path:
            junk_cci_labels = pd.read_csv(file, header=None)
            junk_cci_labels[0] = junk_cci_labels[0].apply(lambda x: x+graph.number_of_nodes())
            junk_cci_labels[1] = junk_cci_labels[1].apply(lambda x: x+graph.number_of_nodes())
            junk_cci_labels = junk_cci_labels.values.tolist()

            # junk_cci_labels = np.asarray(junk_cci_labels)
            # index = np.random.choice(junk_cci_labels.shape[0], len(junk_cci_labels)*0.5)
            # junk_cci_labels = junk_cci_labels[index].values().tolist()

            test_cci_labels += junk_cci_labels

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
    split_mask = random.sample(range(0, num_pairs), int(0.8*num_pairs))
    train_mask[split_mask] += 1
    vali_mask = torch.where(train_mask>0, torch.full_like(train_mask, 0), torch.full_like(train_mask, 1))


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

    load_mouse_mammary_gland(params)
