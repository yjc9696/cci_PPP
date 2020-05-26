#! /usr/local/bin/python
#! -*- encoding:utf-8 -*-
from pathlib import Path
import pandas as pd
import numpy as np
import random
import os

def generate_gt(clusters, dataset):

    cci_labels_gt_path = '{}/mouse_small_intestine_1189_cci_labels_gt_{}_{}.csv'
    cci_labels_junk_path = '{}/mouse_small_intestine_1189_cci_labels_junk_{}_{}.csv'

    data_path = 'mouse_small_intestine_1189_data.csv'
    type_path = 'mouse_small_intestine_1189_cellcluster.csv'

    cci_path = 'mouse_small_intestine_1189_cluster_cluster_interaction_combined.csv'
    ligand_receptor_pair_path = 'mouse_ligand_receptor_pair.csv'

    # prepare data and cell2type
    df = pd.read_csv(data_path, index_col=0)  # (gene, cell)
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
    pair1_mask = ligand + receptor
    pair2_mask = receptor + ligand

    mp = dict()
    for m in range(len(cci)):
        id1 = cci.iloc[m]['cluster1']
        id2 = cci.iloc[m]['cluster2']
        if (id1 not in clusters) or (id2 not in clusters):
            continue

        print(f'cluster: {id1}, {id2}')
        df1 = df[df['type'] == id1]
        df2 = df[df['type'] == id2]
        print(f'total pairs: {len(df1)*len(df2)}')

        # assert len(df1) > 0, f"the cluster {id1} doesn't appear in the dataset."
        # assert len(df2) > 0, f"the cluster {id2} doesn't appear in the dataset."

        cur_cci = []
        cur_cci_junk = []
        for i in range(len(df1)):
            for j in range(len(df2)):
                pair1 = df1.iloc[i][pair1_mask].fillna(0)
                pair1.index = list(range(len(pair1)))
                pair2 = df2.iloc[j][pair2_mask].fillna(0)
                pair2.index = list(range(len(pair2)))
                pair = pd.concat([pair1, pair2], 1)

                flag = False
                for k in range(len(pair)):
                    if pair.iloc[k][0] > 0 and pair.iloc[k][1] > 0:
                        cur_cci.append([df1.iloc[i]['id'], df2.iloc[j]['id'], 1])
                        mp[df1.iloc[i]['id']] = df2.iloc[j]['id']
                        flag = True
                        break
                if not flag and df1.iloc[i]['id'] not in mp:
                    cur_cci_junk.append([df1.iloc[i]['id'], df2.iloc[j]['id'], 0])

        with open(cci_labels_gt_path.format(dataset, id1, id2), 'w', encoding='utf-8') as f:
            print(f"cur cci {len(cur_cci)}")
            for cci_label in cur_cci:
                f.write(f"{int(cci_label[0])},{int(cci_label[1])},{int(cci_label[2])}\r\n")
        with open(cci_labels_junk_path.format(dataset, id1, id2), 'w', encoding='utf-8') as f:
            print(f"cur cci junk {len(cur_cci_junk)}")
            for cci_label in cur_cci_junk:
                f.write(f"{int(cci_label[0])},{int(cci_label[1])},{int(cci_label[2])}\r\n")


def generate_junk(clusters, dataset):
    dir_path = Path(os.getcwd())
    dataset_path = dir_path / dataset
    data_files = dataset_path.glob('*gt*.csv')

    mp = dict()
    for file in data_files:
        print(file)
        with open(str(file), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                tri = line.strip().split(',')
                if int(tri[0]) not in mp:
                    mp[int(tri[0])] = set()
                mp[int(tri[0])].add(int(tri[1]))

    data_path = 'mouse_small_intestine_1189_data.csv'
    df = pd.read_csv(data_path, index_col=0)  # (gene, cell)
    df = df.fillna(0)
    df = df.transpose(copy=True)  # (cell, gene)
    df['id'] = range(0, len(df))  # add cell id
    df['id'].astype(int)

    cur_cci_junk = []
    for i in range(len(df)):
        if i > 0:
            break
        for j in range(i+1, len(df)):
            if i in mp and j in mp[i]:
                continue
            else:
                cur_cci_junk.append([i, j, 0])

    print(f'total junk: {len(cur_cci_junk)}')
    cci_labels_junk_path = '{}/mouse_small_intestine_1189_cci_labels_junk.csv'
    with open(cci_labels_junk_path.format(dataset), 'w', encoding='utf-8') as f:
        print(f"cur cci junk {len(cur_cci_junk)}")
        for cci_label in cur_cci_junk:
            f.write(f"{int(cci_label[0])},{int(cci_label[1])},{int(cci_label[2])}\r\n")


if __name__ == "__main__":
    import os
    os.chdir('/Users/yryang/Desktop/code/cci/data/cell_cell_interaction/')
    print(os.getcwd())
    test_cluster = [5,6,10,11,12,15]


    train_cluster = [1, 2,3,4,7,8,9,13,14,16,17,18,19]

    # test_dataset
    # generate_gt(test_cluster, dataset='test_dataset')
    generate_junk(test_cluster, dataset='test_dataset')

    # train_dataset
    # generate_gt(train_cluster, dataset='train_dataset')
    # generate_junk(train_cluster, dataset='train_dataset')