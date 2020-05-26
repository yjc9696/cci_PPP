#! /usr/local/bin/python
# ! -*- encoding:utf-8 -*-
from pathlib import Path
import pandas as pd
import numpy as np
import random
import os
import argparse

proj_path = Path(__file__).parent.resolve().parent.resolve()
print(proj_path)
data_path = proj_path / 'data' / 'PPP'


def read_csv(data_file, dataset):
    cci_labels_gt_path = data_path / dataset / 'PPP_gt.csv'
    cci_labels_junk_path = data_path / dataset / 'PPP_junk.csv'
    edge_list_path = data_path / data_file

    df = pd.read_csv(edge_list_path, header=None)
    generate_gt(df, dataset)


def generate_gt(df, dataset):
    cci_labels_gt_path = data_path / dataset / 'PPP_gt.csv'
    cci_labels_junk_path = data_path / dataset / 'PPP_junk.csv'
    edge_list = [];
    for indexs in df.index:
        rowData = df.loc[indexs].values[0:2]
        rowData = rowData.tolist()
        edge_list.append(rowData)
    nodes = [];
    for edge in edge_list:
        if edge[0] not in nodes:
            nodes.append(edge[0])
        if edge[1] not in nodes:
            nodes.append(edge[1])
    nodes.sort()
    gene2id = {gene: idx for idx, gene in enumerate(nodes)}
    cur_cci_gt = []
    cur_cci_junk = []
    for indexs in df.index:
        rowData = df.loc[indexs].values[0:2]
        rowData = rowData.tolist()
        p1 = rowData[0]
        p2 = rowData[1]
        choice = random.randint(0, 1)
        cur_cci_gt.append([p1, p2])
        if choice:
            a, c = find_junk(p1, nodes, edge_list, cur_cci_junk)
        else:
            a, c = find_junk(p2, nodes, edge_list, cur_cci_junk)
        cur_cci_junk.append([a, c])

    with open(cci_labels_gt_path, 'w', encoding='utf-8') as f:
        print(f"cur cci {len(cur_cci_gt)}")
        for cci_label in cur_cci_gt:
            f.write(f"{int(cci_label[0])},{int(cci_label[1])}\r\n")

    with open(cci_labels_junk_path, 'w', encoding='utf-8') as f:
        print(f"cur cci junk {len(cur_cci_junk)}")
        for cci_label in cur_cci_junk:
            f.write(f"{int(cci_label[0])},{int(cci_label[1])}\r\n")

        # with open(cci_labels_junk_path.format(dataset, id2, type1), 'w', encoding='utf-8') as f:
        #     print(f"cur cci junk {len(cur_cci_junk_b2d)}")
        #     for cci_label in cur_cci_junk_b2d:
        #         f.write(f"{int(cci_label[0])},{int(cci_label[1])},{int(cci_label[2])}\r\n")


def find_junk(a, nodes, edge_list, cur_cci_junk):
    """


    """
    c = random.choice(nodes)
    while [a, c] in nodes or [a, c] in cur_cci_junk:
        c = random.choice(nodes)
    return a, c


def clean_cross_data(df1, df2):
    df3 = pd.DataFrame()
    nodes = []
    for indexs in df1.index:
        rowData = df1.loc[indexs].values[0:2]
        rowData = rowData.tolist()
        nodes.append(rowData[0])
        nodes.append(rowData[1])
    for indexs in df2.index:
        rowData = df2.loc[indexs].values[0:2]
        rowData = rowData.tolist()
        if (rowData[0] not in nodes) and (rowData[1] not in nodes):
            df3 = df3.append(df2.loc[indexs])
    return df3


if __name__ == "__main__":
    import os

    # python ./code/generate_PPP.py --dataset dataset_all_cross --data_path PP-Pathways_ppi.csv --cross_data 1 --train_rate 0.01
    parser = argparse.ArgumentParser(description='GraphSAGE')
    parser.add_argument("--random_seed", type=int, default=10086)
    parser.add_argument("--dataset", type=str, default='dataset')
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--train_rate", type=float, default=0.1)
    parser.add_argument("--cross_data", type=int, default=0)
    params = parser.parse_args()
    random.seed(params.random_seed)
    np.random.seed(params.random_seed)
    if params.data_path == None:
        # test_dataset
        print('begin test generate:')
        read_csv("PP-9001~10000.csv", dataset='test_' + params.dataset)

        # train_dataset
        print('begin train  generate:')
        read_csv('PP-1~9000.csv', dataset='train_' + params.dataset)
    else:
        edge_list_path = data_path / params.data_path
        df = pd.read_csv(edge_list_path, header=None)
        lens = len(df)
        train_size = int(lens * params.train_rate)
        df1 = df[0:train_size]
        df2 = df[train_size:lens]

        print('begin test generate:')
        generate_gt(df1, dataset='test_' + params.dataset)
        if params.cross_data == 1:
            print('begin clean_cross_data:')
            df2 = clean_cross_data(df1, df2)
        print('begin train generate:')
        generate_gt(df2, dataset='train_' + params.dataset)
