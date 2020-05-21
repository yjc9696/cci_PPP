#! /usr/local/bin/python
# ! -*- encoding:utf-8 -*-
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
        type1 = cci.iloc[m]['cluster1']
        type2 = cci.iloc[m]['cluster2']
        if (type1 not in clusters) or (type2 not in clusters):
            continue

        print(f'cluster: {type1}, {type2}')
        df1 = df[df['type'] == type1]
        df2 = df[df['type'] == type2]
        print(f'ideal total pairs: {len(df1) * len(df2)}')

        # assert len(df1) > 0, f"the cluster {type1} doesn't appear in the dataset."
        # assert len(df2) > 0, f"the cluster {type2} doesn't appear in the dataset."

        cur_cci = []
        cur_cci_junk_a2c = []
        cur_cci_junk_b2d = []
        num = 0
        choice = random.randint(0, 1)
        for i in range(len(df1)):
            for j in range(len(df2)):
                num += 1
                pair1 = df1.iloc[i][pair1_mask].fillna(0)  # series
                pair2 = df2.iloc[j][pair2_mask].fillna(0)
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
                # import pdb;pdb.set_trace()

                if len(pos_mask) > 0:
                    cur_cci.append([df1.iloc[i]['id'], df2.iloc[j]['id'], 1])
                    mp[df1.iloc[i]['id']] = df2.iloc[j]['id']
                    if choice:
                        a, c = find_junk(df, df1.iloc[i]['id'], pair2_mask, pos_mask, clusters, type1, type2)
                    else:
                        # b, d = find_junk(df, df2.iloc[j]['id'], pair1_mask, pos_mask, clusters)
                        a, c = find_junk(df, df2.iloc[j]['id'], pair1_mask, pos_mask, clusters, type1, type2)
                    cur_cci_junk_a2c.append([a, c, 0])
                    # cur_cci_junk_b2d.append([b, d, 0])

        with open(cci_labels_gt_path.format(dataset, type1, type2), 'w', encoding='utf-8') as f:
            print(f"cur cci {len(cur_cci)}")
            for cci_label in cur_cci:
                f.write(f"{int(cci_label[0])},{int(cci_label[1])},{int(cci_label[2])}\r\n")

        with open(cci_labels_junk_path.format(dataset, type1, type2), 'w', encoding='utf-8') as f:
            print(f"cur cci junk {len(cur_cci_junk_a2c)}")
            for cci_label in cur_cci_junk_a2c:
                f.write(f"{int(cci_label[0])},{int(cci_label[1])},{int(cci_label[2])}\r\n")

        # with open(cci_labels_junk_path.format(dataset, id2, type1), 'w', encoding='utf-8') as f:
        #     print(f"cur cci junk {len(cur_cci_junk_b2d)}")
        #     for cci_label in cur_cci_junk_b2d:
        #         f.write(f"{int(cci_label[0])},{int(cci_label[1])},{int(cci_label[2])}\r\n")


def find_junk(df, id1, pair_mask, pos_mask, clusters, type1, type2):
    """

    :param df: df with id type  (cell*gene)
    :param id1: id number
    :param pair_mask: gene mask
    :param pos_mask: the position of gene should be 0
    :return: id1, junk_id
    """

    mask = [pair_mask[i] for i in pos_mask]
    df_junk = df[mask]
    real_idx = list()

    df_junk = df_junk.applymap(lambda x: 1 if x > 0 else 0)

    s = df_junk.apply(lambda x: x.sum(), axis=1)

    for i in range(len(df_junk)):
        cur_id = s.idxmin()
        s.loc[cur_id] = 1e5
        cur_type = int(df.loc[cur_id]['type'])
        if cur_type in clusters and cur_type != type1 and cur_type != type2:
            real_idx.append(i)
            if len(real_idx) > 5:
                break
    rand = random.randint(0, len(real_idx) - 1)

    return id1, int(real_idx[rand])


if __name__ == "__main__":
    cur_path = Path(__file__).parent.resolve()
    import os

    random.seed(10086)
    np.random.seed(10086)

    os.chdir(cur_path)
    print(os.getcwd())
    # 2
    test_cluster = [5, 9, 10, 15, 18, 19]
    train_cluster = [1, 2, 3, 4, 6, 7, 8, 11, 12, 13, 14, 16, 17]
    # test_dataset
    print('begin test cluster generate:')
    generate_gt(test_cluster, dataset='test_dataset4')

    # train_dataset
    print('begin train cluster generate:')
    generate_gt(train_cluster, dataset='train_dataset4')

    # 3
    test_cluster = [2, 4, 6, 8, 11, 12]
    train_cluster = [1, 3, 5, 7, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20]

    # test_dataset
    print('begin test cluster generate:')
    generate_gt(test_cluster, dataset='test_dataset3')

    # # train_dataset
    print('begin train cluster generate:')
    generate_gt(train_cluster, dataset='train_dataset3')
