#! /usr/local/bin/python
# ! -*- encoding:utf-8 -*-
# python codes/data_analyze/all_pairs.py
# 生成所有pair的分析数据，格式为
# cell1, cell2, cluster1, cluster2, cci(1: enrich, 0: depleted, -1: unknown),
# num_masks, scores, max_score, cell1_name, cell2_name

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
from multiprocessing import Pool, Process, Lock
import threading


def generate_gt(params):
    proj_path = Path(__file__).parent.resolve().parent.resolve().parent.resolve()
    mouse_data_path = proj_path / 'data' / params.data_dir  # small_intestine
    dataset_path = mouse_data_path / params.dataset  # train_dataset
    clusters = params.clusters

    ligand_receptor_path = dataset_path / params.ligand_receptor_gene

    cell2cluster_path = dataset_path / params.cell_cluster
    cell_data_path = dataset_path / params.cell_data

    cluster_cluster_interaction_enriched = dataset_path / params.cluster_cluster_interaction_enriched
    cluster_cluster_interaction_depleted = dataset_path / params.cluster_cluster_interaction_depleted

    # store the result
    analyze_path = dataset_path / params.analyze_file


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

    cci = pd.read_csv(cluster_cluster_interaction_enriched, header=0, index_col=0)
    enriched = set()
    for i, row in cci.iterrows():
        enriched.add((row['cluster1'], row['cluster2']))
        enriched.add((row['cluster2'], row['cluster1']))

    cci = pd.read_csv(cluster_cluster_interaction_depleted, header=0, index_col=0)
    depleted = set()
    for i, row in cci.iterrows():
        depleted.add((row['cluster1'], row['cluster2']))
        depleted.add((row['cluster2'], row['cluster1']))

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

    def one_process(begin, end, lock):
        
        df1 = df_mask1.iloc[begin: end]
        df2 = df_mask2
        print(f'begin generate {begin}->{end}, {len(df1)*len(df2)}')
        cur_cci = []

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
                pair[3] = pair[0]*pair[1]

                pair[2] = np.zeros(len(pair)).astype(int)
                pair[2] = pair[2].where(pair[0] == 0, 1)  # 将a中非0的找出来
                pair[2] = pair[2].where(pair[1] != 0, 0)  # 将b中为0的改回去
                tmp = np.array(pair[2])
                pos_mask = np.where(tmp > 0)[0].tolist()

                num_mask = len(pos_mask)
                score = pair[3].sum()
                max_score = pair[3].max()

                type1 = rowi['type']
                type2 = rowj['type']
                if (type1, type2) in enriched:
                    interaction = 1
                elif (type1, type2) in depleted:
                    interaction = 0
                else:
                    interaction = -1

                # id1, id2, type1, type2, cci, mask_num, score, max_score, cell_name1, cell_name2
                cur_cci.append([rowi['id'], rowj['id'], rowi['type'], rowj['type'], interaction,
                                num_mask, score, max_score, rowi['cell'], rowj['cell']])

        try:
            lock.acquire()
            print(f'{begin} -> {end} is done.')
            with open(analyze_path, 'a+', encoding='utf-8') as f:
                for line in cur_cci:
                    f.write(f'{int(line[0])},{int(line[1])},{int(line[2])},{int(line[3])},{int(line[4])},'
                            f'{int(line[5])},{line[6]},{line[7]},"{line[8]}","{line[9]}"\n')
        except Exception as e:
            print(e)
        finally:
            lock.release()



    print('Parent process %s.' % os.getpid())
    with open(analyze_path, 'w', encoding='utf-8') as f:
        f.write('id1,id2,type1,type2,cci,mask_num,score,max_score,cell_name1,cell_name2\n')
    lock = Lock()
    p_obj = []
    num = len(df)
    # num = 40
    m = 20
    for i in range(m):
        begin = num // m * i
        end = num // m * (i+1)
        if i == m-1:
            end = num
        # begin included, end not included
        p = Process(target=one_process, args=(begin, end, lock))
        p_obj.append(p)
    print('Waiting for all subprocesses done...')
    for i in p_obj:
        i.start()
    for i in p_obj:
        i.join()
    print('All subprocesses done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze marrow gene')
    parser.add_argument("--random_seed", type=int, default=10086)
    parser.add_argument("--data_dir", type=str, default='mouse_small_intestine',
                        help="root path of the data dir")
    parser.add_argument("--dataset", type=str, default='train_dataset',
                        help=" dataset")
    # parser.add_argument("--dataset", type=str, default='test_dataset',
    #                     help="test dataset")
    parser.add_argument("--cell_data", type=str, default='mouse_small_intestine_1189_data.csv',
                        help="cell data gene")
    parser.add_argument("--cell_cluster", type=str, default='mouse_small_intestine_1189_cellcluster.csv',
                        help="cell belongs to which cluster")
    parser.add_argument("--cluster_cluster_interaction_enriched", type=str,
                        default='mouse_small_intestine_1189_cluster_cluster_interaction_enriched.csv',
                        help="cluster - cluster interaction enriched")
    parser.add_argument("--cluster_cluster_interaction_depleted", type=str,
                        default='mouse_small_intestine_1189_cluster_cluster_interaction_depleted.csv',
                        help="cluster - cluster interaction depleted")
    parser.add_argument("--ligand_receptor_gene", type=str, default='mouse_ligand_receptor_pair.csv',
                        help="cluster - cluster interaction depleted")
    # parser.add_argument("--clusters", nargs='+', type=int, default=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45],
    #                     help="cluster used to train")
    parser.add_argument("--clusters", nargs='+', type=int, default=[1, 3, 5, 7, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20],
                        help="cluster used to train")
    # parser.add_argument("--clusters", nargs='+', type=int, default=[2, 4, 6, 8, 11, 12],
    #                     help="cluster used to test")

    parser.add_argument("--analyze_file", type=str, default='mouse_small_intestine_1189_analyze.csv',
                        help="the file to store the result")

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



