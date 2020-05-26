#! /usr/local/bin/python
# ! -*- encoding:utf-8 -*-
from pathlib import Path
import pandas as pd
import numpy as np
import random
import os


def analyze(clusters, dataset):
    cci_labels_gt_path = '{}/mouse_small_intestine_1189_cci_labels_gt_{}_{}.csv'
    cci_labels_junk_path = '{}/mouse_small_intestine_1189_cci_labels_junk_{}_{}.csv'

    data_path = 'mouse_small_intestine_1189_data.csv'
    type_path = 'mouse_small_intestine_1189_cellcluster.csv'

    cci_path = 'mouse_small_intestine_1189_cluster_cluster_interaction_combined.csv'
    ligand_receptor_pair_path = 'mouse_ligand_receptor_pair.csv'

    # prepare data and cell2type
    df = pd.read_csv(data_path, index_col=0)  # (gene, cell)
    genes = set(df.index.tolist())
    df = df.fillna(0)
    df = df.transpose(copy=True)  # (cell, gene)
    df['id'] = range(0, len(df))  # add cell id
    df['id'].astype(int)
    cell2type = pd.read_csv(type_path, index_col=0)
    cell2type.columns = ['cell', 'type']
    assert cell2type['cell'].tolist() == df.index.tolist()
    df['type'] = cell2type['type'].tolist()

    # prepare cell cell interaction
    cci = pd.read_csv(cci_path, header=0, index_col=0)

    # prepare ligand receptor pair
    lcp = pd.read_csv(ligand_receptor_pair_path, header=0, index_col=0)
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

    mp = dict()
    for m in range(len(cci)):
        id1 = cci.iloc[m]['cluster1']
        id2 = cci.iloc[m]['cluster2']
        if (id1 not in clusters) or (id2 not in clusters):
            continue

        print(f'cluster: {id1}, {id2}')
        df1 = df[df['type'] == id1]
        df2 = df[df['type'] == id2]
        print(f'ideal total pairs: {len(df1) * len(df2)}')





if __name__ == "__main__":
    cur_path = Path(__file__).parent.resolve()
    import os

    random.seed(10086)

    os.chdir(cur_path)
    print(os.getcwd())
    test_cluster = [5, 9, 10, 15, 18, 19]
    train_cluster = [1, 2, 3, 4, 6, 7, 8, 11, 12, 13, 14, 16, 17]

    # test_dataset
    print('begin test cluster analyze:')
    analyze(test_cluster, dataset='test_dataset2')

    # train_dataset
    print('begin train cluster analyze:')
    analyze(train_cluster, dataset='train_dataset2')
