#! /usr/local/bin/python
# ! -*- encoding:utf-8 -*-
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import random, time
import os
import sys
sys.path.append(str(Path(__file__).parent.resolve().parent.resolve()))
# print(sys.path)
from torchlight import set_seed
from multiprocessing import Pool, Process
import threading


def generate_gt(params):
    proj_path = Path(__file__).parent.resolve().parent.resolve().parent.resolve()
    mouse_data_path = proj_path / 'data' / params.data_dir # small_intestine
    dataset_path = mouse_data_path / params.dataset # train_dataset
    clusters = params.clusters

    ligand_receptor_path = dataset_path / params.ligand_receptor_gene

    cell2cluster_path = dataset_path / params.cell_cluster
    cell_data_path = dataset_path / params.cell_data

    cluster_cluster_interaction_enriched = dataset_path / params.cluster_cluster_interaction_enriched
    cluster_cluster_interaction_depleted = dataset_path / params.cluster_cluster_interaction_depleted

    cci_labels_gt_path = '{}/data/mouse_small_intestine_1189_cci_labels_gt_{}_{}.csv'
    cci_labels_junk_path = '{}/data/mouse_small_intestine_1189_cci_labels_junk_{}_{}.csv'

    pos_or_neg = params.pos_or_neg

    # prepare data and cell2type
    df = pd.read_csv(cell_data_path, index_col=0)  # (gene, cell)
    genes = set(df.index.tolist())
    df = df.fillna(0)
    df = df.transpose(copy=True)  # (cell, gene)
    
    df['id'] = range(0, len(df))  # add cell id
    df['id'].astype(int)
    cell2type = pd.read_csv(cell2cluster_path, index_col=0)
    cell2type.columns = ['cell', 'type']
    assert cell2type['cell'].tolist() == df.index.tolist()
    df['type'] = cell2type['type'].tolist()
    df['cell'] = df.index.tolist()

    # prepare cell cell interaction
    if pos_or_neg == 'pos':
        cci = pd.read_csv(cluster_cluster_interaction_enriched, header=0, index_col=0)
    else:
        cci = pd.read_csv(cluster_cluster_interaction_depleted, header=0, index_col=0)
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

    df_mask1 = df[pair1_mask]
    df_mask1.is_copy = None
    df_mask1['type'] = df['type'].tolist()
    df_mask1['id'] = df['id'].tolist()
    df_mask1['cell'] = df['cell'].tolist()
    df_mask2 = df[pair2_mask]
    df_mask2.is_copy = None
    df_mask2['type'] = df['type'].tolist()
    df_mask2['id'] = df['id'].tolist()
    df_mask2['cell'] = df['cell'].tolist()

    mp = dict()
    def one_process(m):
        scores = list()
        type1 = cci.iloc[m]['cluster1']
        type2 = cci.iloc[m]['cluster2']
        print(type1, type2)
        if (type1 not in clusters) or (type2 not in clusters):
            return

        print(f'cluster: {type1}, {type2}')
        df1 = df_mask1[df_mask1['type'] == type1]
        df2 = df_mask2[df_mask2['type'] == type2]
        print(f'ideal total pairs: {len(df1) * len(df2)}')

        # assert len(df1) > 0, f"the cluster {type1} doesn't appear in the dataset."
        # assert len(df2) > 0, f"the cluster {type2} doesn't appear in the dataset."

        cur_cci = []
        cur_cci_junk_a2c = []
        cur_cci_junk_b2d = []
        num = 0
        choice = random.randint(0, 1)
        for i, rowi in df1.iterrows():
            for j, rowj in df2.iterrows():
                pair1 = rowi.fillna(0)  # series
                pair2 = rowj.fillna(0)
                pair1 = pair1.drop(['type', 'id', 'cell'])
                pair2 = pair2.drop(['type', 'id', 'cell'])
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
                # pair[3] = pair[0]*pair[1]
#                 scores.append(len(pos_mask))
        #         scores.append(pair[3].sum())
        # scores_df = pd.DataFrame(scores)
        # print(scores_df.describe())
                if len(pos_mask) > 10:
                    if pos_or_neg == 'pos':
                    # id1, id2, pos_or_neg, mask_num, type1, type2
                        cur_cci.append([rowi['id'], rowj['id'], 1, len(pos_mask), type1, type2])
                    else:
                        cur_cci.append([rowi['id'], rowj['id'], 0, len(pos_mask), type1, type2])
                    # if choice:
                    #     a, c = find_junk(df, rowi['id'], pair2_mask, pos_mask, clusters, type1, type2)
                    # else:
                    #     # b, d = find_junk(df, df2.iloc[j]['id'], pair1_mask, pos_mask, clusters)
                    #     a, c = find_junk(df, rowj['id'], pair1_mask, pos_mask, clusters, type1, type2)
                    # cur_cci_junk_a2c.append([a, c, 0])
                    # # cur_cci_junk_b2d.append([b, d, 0])
                    num += 1
        if pos_or_neg == 'pos':
            with open(cci_labels_gt_path.format(dataset_path, type1, type2), 'w', encoding='utf-8') as f:
                print(f"cur cci {type1} {type2} -> {len(cur_cci)}")
                for cci_label in cur_cci:
                    f.write(f'{(cci_label[0])},{(cci_label[1])},{int(cci_label[2])},{int(cci_label[3])},{int(cci_label[4])},{int(cci_label[5])}\r\n')
        else:
            with open(cci_labels_junk_path.format(dataset_path, type1, type2), 'w', encoding='utf-8') as f:
                print(f"cur junk cci {type1} {type2} -> {len(cur_cci)}")
                for cci_label in cur_cci:
                    f.write(f'{(cci_label[0])},{(cci_label[1])},{int(cci_label[2])},{int(cci_label[3])},{int(cci_label[4])},{int(cci_label[5])}\r\n')
        # with open(cci_labels_junk_path.format(dataset_path, type1, type2), 'w', encoding='utf-8') as f:
        #     print(f"cur cci junk {type1} {type2} -> {len(cur_cci_junk_a2c)}")
        #     for cci_label in cur_cci_junk_a2c:
        #         f.write(f'{(cci_label[0])},{(cci_label[1])},{int(cci_label[2])}\r\n')

    print('Parent process %s.' % os.getpid())
    p_obj = []
    for i in range(len(cci)):
        p = Process(target=one_process, args=(i,))
        p_obj.append(p)
    print('Waiting for all subprocesses done...')
    for i in p_obj:
        i.start()
    for i in p_obj:
        i.join()
    print('All subprocesses done.')


# def find_junk(df, id1, pair_mask, pos_mask, clusters, type1, type2):
#     """

#     :param df: df with id type cell  (cell*gene)
#     :param id1: id number
#     :param pair_mask: gene mask
#     :param pos_mask: the position of gene should be 0
#     :return: cell1, junk_cell
#     """
    
#     # print(f'junk func: {type1}, {type2}')
#     mask = [pair_mask[i] for i in pos_mask]
#     df_junk = df[mask]
#     real_idx = list()

#     df_junk = df_junk.applymap(lambda x: 1 if x > 0 else 0)

#     s = df_junk.apply(lambda x: x.sum(), axis=1)
#     # 细胞sum的含量，按cellid依次排列
#     s = s.tolist()
#     new_s = list()
#     idx2id = dict()
#     idx = 0
#     cellid = 0
#     for cur_type in df['type'].tolist():
#         if cur_type in clusters and cur_type != type1 and cur_type != type2:
#             idx2id[idx] = cellid
#             idx += 1
#             new_s.append(s[cellid])
#         cellid += 1
#     s = np.array(new_s)
#     assert len(s) == len(idx2id), 'error with junk'
#     re_ne = 20
#     ids = np.argpartition(s, re_ne)[:re_ne]
#     cur_id = ids[np.random.randint(len(ids), size=1)[0]]
#     id2 = idx2id[cur_id]
#     # print('end junk')
#     return id1, id2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze marrow gene')
    parser.add_argument("--random_seed", type=int, default=10086)
    parser.add_argument("--data_dir", type=str, default='mouse_small_intestine',
                        help="root path of the data dir")
    parser.add_argument("--dataset", type=str, default='train_dataset',
                        help=" dataset")
    # parser.add_argument("--dataset", type=str, default='test_dataset',
    #                     help="test dataset")
    parser.add_argument("--pos_or_neg", type=str, default='pos',
                        help="generate positive pairs")
    parser.add_argument("--cell_data", type=str, default='mouse_small_intestine_1189_data.csv',
                        help="cell data gene")
    parser.add_argument("--cell_cluster", type=str, default='mouse_small_intestine_1189_cellcluster.csv',
                        help="cell belongs to which cluster")
    parser.add_argument("--cluster_cluster_interaction_enriched", type=str, default='mouse_small_intestine_1189_cluster_cluster_interaction_enriched.csv',
                        help="cluster - cluster interaction enriched")
    parser.add_argument("--cluster_cluster_interaction_depleted", type=str, default='mouse_small_intestine_1189_cluster_cluster_interaction_depleted.csv',
                        help="cluster - cluster interaction depleted")
    parser.add_argument("--ligand_receptor_gene", type=str, default='mouse_ligand_receptor_pair.csv',
                        help="cluster - cluster interaction depleted")
    # parser.add_argument("--clusters", nargs='+', type=int, default=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45],
    #                     help="cluster used to train")
    parser.add_argument("--clusters", nargs='+', type=int, default=[1, 3, 5, 7, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20],
                        help="cluster used to train")
    # parser.add_argument("--clusters", nargs='+', type=int, default=[2, 4, 6, 8, 11, 12],
    #                     help="cluster used to test")

    params = parser.parse_args()
    print(params)

    set_seed(params.random_seed)
    # split the small intestine
    # test_cluster = [2, 4, 6, 8, 11, 12]
    # train_cluster = [1, 3, 5, 7, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20]

    # small intestine
    # train_cluster = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    # test_cluster = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45]

    generate_gt(params)



