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

from .dataset import TrainSet



def load_mouse_mammary_gland(params):
    random_seed = params.random_seed
    dense_dim = params.dense_dim
    # root = params.root
    train = params.dataset

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
    pair1_mask = ligand + receptor
    pair2_mask = receptor + ligand
    # assert(len(ligand) == len(receptor), "ligand num should match receptor num.")


    cci_labels = []
    cci_of_1_num = 0

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
        # print(df.head())
        print(f'Nonzero Ratio: {df.fillna(0).astype(bool).sum().sum() / df.size * 100:.2f}%')

        # choose the pairs that have ligand and receptor gene
        cci_path = mouse_data_path / f'mouse_{tissue}_{num}_cluster_cluster_interaction_combined.csv'
        cci = pd.read_csv(cci_path, header=0, index_col=0, dtype=str)
        cci['type1'] = cci['cluster1'].map(label2id)
        cci['type2'] = cci['cluster2'].map(label2id)

        # stores the pairs that have relation
        cci_labels_gt_path = statistics_path / (f'mouse_{tissue}_{num}_cci_labels_gt.csv')
        
        if not cci_labels_gt_path.exists():
            for i in range(len(df)):
                print(i)
                print(cci_of_1_num)

                cur_cci = []
                
                for j in range(i, len(df)):
                    if i != j:
                        pair1 = df.iloc[i][pair1_mask].fillna(0)
                        pair1.index = list(range(len(pair1)))
                        pair2 = df.iloc[j][pair2_mask].fillna(0)
                        pair2.index = list(range(len(pair2)))
                        pair = pd.concat([pair1, pair2], 1)
                        for k in range(len(pair)):
                            if pair.iloc[k][0] > 0 and pair.iloc[k][1] > 0:
                                type1 = cell2type.iloc[i]['id']
                                type2 = cell2type.iloc[j]['id']
                                for m in range(len(cci)):
                                    id1 = cci.iloc[m]['type1']
                                    id2 = cci.iloc[m]['type2']

                                    if (type1 == id1 and type2 == id2) or (type1 == id2 and type2 == id1):
                                        cci_labels.append([i, j, 1])
                                        cur_cci.append([i, j, 1])
                                        cci_of_1_num += 1
                                        break
                                break

                with open(cci_labels_gt_path, 'a+', encoding='utf-8') as f:
                    print(f"cur cci {len(cur_cci)}")
                    for cci_label in cur_cci:
                        f.write(f"{cci_label[0]},{cci_label[1]},{cci_label[2]}\r\n")

            with open('total'+cci_labels_gt_path, 'w', encoding='utf-8') as f:
                for cci_label in cci_labels:
                    f.write(f"{cci_label[0]},{cci_label[1]},{cci_label[2]}\r\n")
        
        cci_labels = pd.read_csv(cci_labels_gt_path, header=None)
        cci_labels[0] = cci_labels[0].apply(lambda x: x+graph.number_of_nodes())
        cci_labels[1] = cci_labels[1].apply(lambda x: x+graph.number_of_nodes())
        cci_labels = cci_labels.values.tolist()[:1000]
        #debug
        for i in range(10000):
            j = int((random.random())%1120 + 5)
            cci_labels.append([i%1180 + 2+num_genes, j + num_genes, 0])


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
    # use weighted gene_feat as cell_feat
    cell_feat = sparse_feat.dot(gene_feat)
    gene_feat = torch.from_numpy(gene_feat)  # use shared storage
    cell_feat = torch.from_numpy(cell_feat)

    features = torch.cat([gene_feat, cell_feat], dim=0).type(torch.float)
    # features = torch.FloatTensor(graph.number_of_nodes(), params.dense_dim).normal_()
    
    
    
    # 3. then create masks for different purposes.

    print(f"load cci ground truth done: {len(cci_labels)}")

    num_cells = graph.number_of_nodes() - num_genes
    

    # cci_path = mouse_data_path / f'mouse_{tissue}_{num}_cluster_cluster_interaction_combined.csv'
    # cci = pd.read_csv(cci_path, header=0, index_col=0, dtype=str)
    # # choose two types as a, b
    # type1, type2 = label2id[cci.iloc[1][0]], label2id[cci.iloc[1][1]]

    # # generate pair
    # cci_labels = []
    # cci_of_1_num = 0
    # for i, label1 in enumerate(labels):
    #     for j, label2 in enumerate(labels):
    #         if type1 == label1 and type2 == label2:
    #             cci_labels.append([i+num_genes, j+num_genes, 1])
    #             cci_of_1_num += 1
    #         elif type1 == label2 and type2 == label1:
    #             cci_labels.append([i + num_genes, j + num_genes, 1])
    #             cci_of_1_num += 1
    #         elif type1 == label1 or type2 == label2 or type1 == label2 or type2 == label1:
    #             if random.random() < 0.01:
    #                 cci_labels.append([i + num_genes, j + num_genes, 0])
    #         else:
    #             pass

    cci_labels = torch.LongTensor(cci_labels)
    num_pairs = len(cci_labels)
    print(f"Total {len(cci_labels)} pairs. A and B pairs are: {cci_of_1_num}")

    train_mask = torch.zeros(num_pairs, dtype=torch.int32)
    test_mask = torch.zeros(num_pairs, dtype=torch.int32)

    # import pdb;pdb.set_trace()
    split_mask = random.sample(range(0, num_pairs), int(0.8*num_pairs))
    train_mask[split_mask] += 1
    test_mask = torch.where(train_mask>0, torch.full_like(train_mask, 0), torch.full_like(train_mask, 1))


    assert train_mask.sum().item() + test_mask.sum().item() == num_pairs
    train_mask = train_mask.type(torch.bool)
    test_mask = test_mask.type(torch.bool)
    # return num_cells, num_genes, num_labels, graph, features, cci_labels, train_mask, test_mask
    return num_cells, num_genes, 2, graph, features, cci_labels, train_mask, test_mask


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
