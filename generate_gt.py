#! /usr/local/bin/python
#! -*- encoding:utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np
import random


def generate_gt(clusters, only_pos=True):


    dataset = 'test_dataset' if only_pos else 'train_dataset'

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
    cell2type = pd.read_csv(type_path, index_col=0, dtype=str)
    cell2type.columns = ['cell', 'type']
    assert cell2type['cell'].tolist() == df.index.tolist()

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
        print(f"mp len: {len(mp)}")

        cur_cci = []
        cur_cci_junk = []
        for i in range(len(df)):
            for j in range(i, len(df)):
                if i != j:
                    pair1 = df.iloc[i][pair1_mask].fillna(0)
                    pair1.index = list(range(len(pair1)))
                    pair2 = df.iloc[j][pair2_mask].fillna(0)
                    pair2.index = list(range(len(pair2)))
                    pair = pd.concat([pair1, pair2], 1)
                    type1 = cell2type.iloc[i]['type']
                    type2 = cell2type.iloc[j]['type']
                    flag = False
                    for k in range(len(pair)):
                        if pair.iloc[k][0] > 0 and pair.iloc[k][1] > 0:
                            if (type1 == id1 and type2 == id2) or (type1 == id2 and type2 == id1):
                                cur_cci.append([i, j, 1])
                                mp[i] = j
                                flag = True
                            break

                    if not flag and i not in mp:
                        cur_cci_junk.append([i, j, 0])

        with open(cci_labels_gt_path.format(dataset, id1, id2), 'w', encoding='utf-8') as f:
            print(f"cur cci: {len(cur_cci)}")
            for cci_label in cur_cci:
                f.write(f"{cci_label[0]},{cci_label[1]},{cci_label[2]}\r\n")
        with open(cci_labels_junk_path.format(dataset, id1, id2), 'w', encoding='utf-8') as f:
            print(f"cur cci junk: {len(cur_cci_junk)}")
            for cci_label in cur_cci_junk:
                f.write(f"{cci_label[0]},{cci_label[1]},{cci_label[2]}\r\n")



if __name__ == "__main__":
    cur_path = Path(__file__).parent.resolve()
    import os
    os.chdir(cur_path)
    print(os.getcwd())
    test_cluster = [5,6,10,11,12,15]
    train_cluster = [1, 2,3,4,7,8,9,13,14,16,17,18,19]
    print('begin test cluster generate:')
    generate_gt(test_cluster, only_pos=True)
    print('begin train cluster generate:')
    generate_gt(train_cluster, only_pos=False)