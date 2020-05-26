import pandas as pd
import dgl
from time import time
import torch
from sklearn.decomposition import PCA
import numpy as np
from torchlight import set_seed


def load_tissue(params=None):
    random_seed = params.random_seed
    dense_dim = params.dense_dim 
    set_seed(random_seed)
    # 400 0.7895
    # 200 0.5117
    # 100 0.3203
    #  50 0.2083
    """
    root = '../data/mammary_gland'
    num = 2915
    data_path = f'{root}/mouse_Mammary_gland{num}_data.csv'
    type_path = f'{root}/mouse_Mammary_gland{num}_celltype.csv'
    """
    data_path = '../data/mouse_data/mouse_brain_2915_data.csv'
    type_path = '../data/mouse_data/mouse_brain_2915_celltype.csv'

    # load celltype file then update labels accordingly
    cell2type = pd.read_csv(type_path, index_col=0)
    cell2type.columns = ['cell', 'type']

    id2label = cell2type['type'].drop_duplicates(keep='first').tolist()
    label2id = {label: idx for idx, label in enumerate(id2label)}
    print(f'{len(id2label)} classes in total')
        
    cell2type['id'] = cell2type['type'].map(label2id)
    assert not cell2type['id'].isnull().any(), 'something wrong about celltype file.'

    # load data file
    data = pd.read_csv(data_path, index_col=0)
    data = data.transpose(copy=True)
    assert cell2type['cell'].tolist() == data.index.tolist()
    print(f'{data.shape[0]} cells, {data.shape[1]} genes.')
    # genes
    id2gene = data.columns.tolist()
    gene2id = {gene: idx for idx, gene in enumerate(id2gene)}

    # construct graph and add nodes and edges
    graph = dgl.DGLGraph()
    start = time()
    # 1. add all genes as nodes
    num_genes = len(id2gene)
    graph.add_nodes(num_genes)
    # maintain a kind of sparse idx for Graph
    row_idx, col_idx = data.to_numpy().nonzero()
    row_idx = row_idx + num_genes
    # 2. add cell nodes and edges
    num_cells = data.shape[0]
    graph.add_nodes(num_cells)
    graph.add_edges(row_idx, col_idx)
    graph.add_edges(col_idx, row_idx)
    print(f'Added {num_cells} nodes and {len(row_idx)} edges.')
    print(f'#Nodes: {graph.number_of_nodes()}, #Edges: {graph.number_of_edges()}.')
    print(data.head())

    # reduce sparse features to dense features
    cell_pca = PCA(n_components=dense_dim, random_state=random_seed)
    cell_pca.fit(data.values)
    cell_feat = cell_pca.transform(data.values)
    cell_feat = torch.FloatTensor(cell_feat)

    gene_pca = PCA(n_components=dense_dim, random_state=random_seed)
    gene_pca.fit(data.T.values)
    gene_feat = gene_pca.transform(data.T.values)
    gene_feat = torch.FloatTensor(gene_feat)

    feat = torch.cat([gene_feat, cell_feat], dim=0)
    # feat = torch.zeros(graph.number_of_nodes(), dense_dim).normal_()

    cell_evr = sum(cell_pca.explained_variance_ratio_) * 100
    gene_evr = sum(gene_pca.explained_variance_ratio_) * 100
    print(f'[PCA] Cell EVR: {cell_evr:.2f}%. Gene EVR: {gene_evr:.2f} %.')
    # generate labels for training and testing
    labels = torch.LongTensor(cell2type['id'].tolist())
    train_mask = torch.zeros(num_cells, dtype=torch.bool)
    train_randidx = torch.randperm(num_cells)[:int(num_cells * 0.8)]
    # generate mask
    train_mask[train_randidx] = True
    test_mask = ~train_mask
    return num_cells, num_genes, graph, feat, labels, train_mask, test_mask


if __name__=='__main__':
    load_tissue()
