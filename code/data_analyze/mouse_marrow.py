import argparse

import pandas as pd
from time import time
import torch
from scipy.sparse import csr_matrix, vstack
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import random
import sys
sys.path.append(str(Path(__file__).parent.resolve().parent.resolve()))
# print(sys.path)
from torchlight import set_seed



def marrow(params):
    proj_path = Path(__file__).parent.resolve().parent.resolve().parent.resolve()
    mouse_data_path = proj_path / 'data'
    train_dataset = mouse_data_path / params.train_dataset
    test_dataset = mouse_data_path / params.test_dataset

    ligand_receptor_path = train_dataset / params.ligand_receptor_gene
    cell2cluster_path = train_dataset / params.cell_cluster
    cell_data_path = train_dataset / params.cell_data
    cluster_cluster_interaction_enriched = train_dataset / params.cluster_cluster_interaction_enriched
    cluster_cluster_interaction_depleted = train_dataset / params.cluster_cluster_interaction_depleted
    cell_cell_interaction_path = train_dataset / params.cell_cell_interaction


    # prepare data
    cell_data = pd.read_csv(cell_data_path, index_col=0).transpose().fillna(0)
    cell2id = dict()
    for idx, cell in enumerate(cell_data.index.tolist()):
        cell2id[cell] = idx
    cell_data.index = list(cell2id.values())

    genes = set(cell_data.columns.tolist())

    # prepare ligand receptor pair
    lcp = pd.read_csv(ligand_receptor_path, header=0, index_col=0)
    ligand = lcp['ligand'].tolist()
    receptor = lcp['receptor'].tolist()

    exist_ligand = list()
    exist_receptor = list()
    for i in range(len(ligand)):
        if ligand[i] in genes and receptor[i] in genes:
            exist_ligand.append(ligand[i])
            exist_receptor.append(receptor[i])

    pair1_mask = exist_ligand + exist_receptor
    pair2_mask = exist_receptor + exist_ligand

    # gene2id
    choose_genes = set(ligand+receptor)
    gene2id = dict()
    for idx, gene in enumerate(choose_genes):
        gene2id[gene] = idx

    # prepare cell2type
    cell2cluster = pd.read_csv(cell2cluster_path, index_col=0)
    cell2cluster['id'] = cell2cluster['cell'].apply(lambda x: cell2id[x])

    # prepare gt cci
    gt_cci = pd.read_csv(cell_cell_interaction_path, index_col=0)
    gt_cci = gt_cci.applymap(lambda x: cell2id[x])

    # cell * gene, gene is the chosen 1000 genes.
    cell_data1 = cell_data[pair1_mask]
    cell_data2 = cell_data[pair2_mask]

    # import pdb; pdb.set_trace()

    all = list()

    for i in range(len(gt_cci)):
        cell1 = gt_cci.iloc[i]['cell1']
        cell2 = gt_cci.iloc[i]['cell2']
        pair1 = cell_data1.iloc[cell1]
        pair2 = cell_data2.iloc[cell2]
        assert np.sum(pair1.index[:len(pair1) // 2] == pair2.index[len(pair2) // 2:]) == len(
            pair2) // 2, "pair mask error"
        pair1.index = list(range(len(pair1)))
        pair2.index = list(range(len(pair2)))
        pair = pd.concat([pair1, pair2], 1)

        pair.columns = [0, 1]
        pair[2] = np.zeros(len(pair)).astype(int)
        pair[2] = pair[2].where(pair[0] == 0, 1)  # 将a中非0的找出来
        pair[2] = pair[2].where(pair[1] != 0, 0)  # 将b中为0的改回去
        tmp = np.array(pair[2])
        pos_mask = np.where(tmp > 0)[0].tolist()
        # print(len(pos_mask))
        # import pdb;pdb.set_trace()
        score = sum(pair1[pos_mask] * pair2[pos_mask])
        all.append(len(pos_mask))
        # import pdb; pdb.set_trace()

    plt.figure()
    plt.scatter(list(range(len(all))), all)
    plt.show()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze marrow gene')
    parser.add_argument("--random_seed", type=int, default=10086)

    parser.add_argument("--train_dataset", type=str, default='validation_dataset',
                        help="train dataset")
    parser.add_argument("--test_dataset", type=str, default='test_dataset',
                        help="test dataset")
    parser.add_argument("--cell_data", type=str, default='mouse_bone_marrow_911_data.csv',
                        help="cell data gene")
    parser.add_argument("--cell_cluster", type=str, default='mouse_bone_marrow_911_cellcluster.csv',
                        help="cell belongs to which cluster")
    parser.add_argument("--cluster_cluster_interaction_enriched", type=str, default='mouse_bone_marrow_911_cluster_cluster_interaction_enriched.csv',
                        help="cluster - cluster interaction enriched")
    parser.add_argument("--cluster_cluster_interaction_depleted", type=str, default='mouse_bone_marrow_911_cluster_cluster_interaction_depleted.csv',
                        help="cluster - cluster interaction depleted")
    parser.add_argument("--ligand_receptor_gene", type=str, default='mouse_ligand_receptor_pair.csv',
                        help="cluster - cluster interaction depleted")
    parser.add_argument("--cell_cell_interaction", type=str, default='mouse_bone_marrow_911_cell_cell_interaction.csv',
                        help="single cell - cell interaction, ground truth")

    params = parser.parse_args()

    set_seed(params.random_seed)
    marrow(params)