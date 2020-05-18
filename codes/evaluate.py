#!/usr/local/bin/python
# -*- coding: utf-8 -*-
"""
to construct the cci throgh the result
"""
from pathlib import Path 
import numpy as np
import pandas as pd
import random
from multiprocessing import Process, Queue
import pickle as pkl
import math

class Evaluate:
    def __init__(self, params):
        """init the class
        
        Args:
            params (args): [description]
        """
        proj_path = Path(__file__).parent.resolve().parent.resolve()
        self.tsne_path = proj_path / 'tmp'
        mouse_data_path = proj_path / 'data' / params.data_dir #small_intestine
        # train_dataset = mouse_data_path / params.train_dataset
        test_dataset = mouse_data_path / params.test_dataset

        cell2cluster_path = test_dataset / params.cell_cluster
        cell_data_path = test_dataset / params.cell_data_path
        # ligand_receptor_path = mouse_data_path / params.ligand_receptor_gene
        cluster2cluster_enriched_path = test_dataset / params.cluster_cluster_interaction_enriched
        cluster2cluster_depleted_path = test_dataset / params.cluster_cluster_interaction_depleted

        # cell * gene
        self.cell_data = pd.read_csv(cell_data_path, index_col=0).fillna(0).transpose()
        # ['cell', 'cluster']
        self.cell2cluster = pd.read_csv(cell2cluster_path, index_col=0)
        assert self.cell2cluster.index.tolist() == self.cell_data.index.tolist(), 'cell data and cell_cluster not match, error'
        
        # 'cluster1', 'cluster2'
        self.cluster2cluster_enriched = pd.read_csv(cluster2cluster_enriched_path, header=0, index_col=0)
        self.cluster2cluster_depleted = pd.read_csv(cluster2cluster_depleted_path, header=0, index_col=0)

        self.clusters = list(set(self.cell2cluster['cluster']))
        # import pdb; pdb.set_trace()
        # print(self.clusters)
        # store the cell id list of some cluster
        self.cluster2cell = dict()
        for i in self.clusters:
            self.cluster2cell[i] = np.where(self.cell2cluster['cluster'] == i)[0]
        
        # store the enriched pairs
        self.cluster_pairs_enriched = list()
        for i in range(len(self.cluster2cluster_enriched)):
            type1 = self.cluster2cluster_enriched.iloc[i]['cluster1']
            type2 = self.cluster2cluster_enriched.iloc[i]['cluster2']
            if type1 in self.clusters and type2 in self.clusters:
                self.cluster_pairs_enriched.append((type1, type2))

        self.cluster_pairs_depleted = list()
        for i in range(len(self.cluster2cluster_depleted)):
            type1 = self.cluster2cluster_depleted.iloc[i]['cluster1']
            type2 = self.cluster2cluster_depleted.iloc[i]['cluster2']
            if type1 in self.clusters and type2 in self.clusters:
                self.cluster_pairs_depleted.append((type1, type2))

        self.cluster_pairs_unknown = list()
        for i in self.clusters:
            for j in self.clusters:
                if i != j: 
                    if (i,j) not in self.cluster_pairs_depleted and (j,i) not in self.cluster_pairs_depleted:
                        if (i,j) not in self.cluster_pairs_enriched and (j,i) not in self.cluster_pairs_enriched:
                            self.cluster_pairs_unknown.append((i,j))
        # print(self.cluster_pairs_enriched)
        # print(self.cluster_pairs_depleted)
        # # print()
        # print(self.cluster_pairs_unknown)
    
    def evaluate_with_percentage(self, cci_predict, cci_gt):
        """evaluate the predict result with 10% percentage
        
        Args:
            cci_gt (np.array pair_num*6): [cell1, cell2, relation, score, cell_id1, cell_id2]
            cci_predict (np.array, pair_num * 1): [labels]
        """
        # indices predicted positive by the model
        nonzero = cci_predict.nonzero()[0]
        # get these pairs
        cci_gt_nonzero = cci_gt[nonzero]
        col = cci_gt_nonzero[:, 4]
        row = cci_gt_nonzero[:, 5]
        

        adj = np.zeros((len(self.cell_data), len(self.cell_data)))
        adj[row, col] = 1
        adj[col, row] = 1
        # import pdb; pdb.set_trace()
        for pair in self.cluster_pairs_enriched:
            type1, type2 = pair
            idx1 = self.cluster2cell[type1]
            idx2 = self.cluster2cell[type2]
            total = math.sqrt(len(idx1) * len(idx2))
            # import pdb; pdb.set_trace()
            row1, _ = adj.nonzero()
            ratio1 = len(set(idx1).intersection(set(row1))) / len(idx1)
            ratio2 = len(set(idx2).intersection(set(row1))) / len(idx2)
            
            print(f'enriched: hit ratio {ratio1*ratio2} ')
        
        for pair in self.cluster_pairs_depleted:
            type1, type2 = pair
            idx1 = self.cluster2cell[type1]
            idx2 = self.cluster2cell[type2]
            total = math.sqrt(len(idx1) * len(idx2))
            # import pdb; pdb.set_trace()
            row1, _ = adj.nonzero()
            ratio1 = len(set(idx1).intersection(set(row1))) / len(idx1)
            ratio2 = len(set(idx2).intersection(set(row1))) / len(idx2)
            
            print(f'depleted: hit ratio {ratio1*ratio2} ')
        
        for pair in self.cluster_pairs_unknown:
            type1, type2 = pair
            idx1 = self.cluster2cell[type1]
            idx2 = self.cluster2cell[type2]
            total = math.sqrt(len(idx1) * len(idx2))
            # import pdb; pdb.set_trace()
            row1, _ = adj.nonzero()
            ratio1 = len(set(idx1).intersection(set(row1))) / len(idx1)
            ratio2 = len(set(idx2).intersection(set(row1))) / len(idx2)
            
            print(f'unknown: hit ratio {ratio1*ratio2} ')

        # # import pdb; pdb.set_trace()
        # for pair in self.cluster_pairs_enriched:
        #     type1, type2 = pair
        #     idx1 = self.cluster2cell[type1]
        #     idx2 = self.cluster2cell[type2]
        #     total = len(idx1) * len(idx2)
        #     # import pdb; pdb.set_trace()
        #     hit = adj[idx1][:, idx2].sum()
        #     print(f'enriched: hit {hit}, total {total}, ratio {hit/total} ')
        
        # for pair in self.cluster_pairs_depleted:
        #     type1, type2 = pair
        #     idx1 = self.cluster2cell[type1]
        #     idx2 = self.cluster2cell[type2]
        #     total = len(idx1) * len(idx2)
        #     hit = adj[idx1][:, idx2].sum()
        #     print(f'depleted: hit {hit}, total {total}, ratio {hit/total} ')
        
        # for pair in self.cluster_pairs_unknown:
        #     type1, type2 = pair
        #     idx1 = self.cluster2cell[type1]
        #     idx2 = self.cluster2cell[type2]
        #     total = len(idx1) * len(idx2)
        #     hit = adj[idx1][:, idx2].sum()
        #     print(f'unknown: hit {hit}, total {total}, ratio {hit/total} ')
        
    def evaluate_with_permuation(self, cci_predict, cci_gt, features, num=10000):
        """evaluate the predicted result with permuation
        
        Args:
            cci_predict (np.array, pair_num * 1): [labels]
            cci_gt (np.array pair_num*6): [cell1, cell2, relation, score, cell_id1, cell_id2]
            features (np.array) (num_genes + num_train_nodes + num_test_nodes) * dim, only need test node fea.
            
        """
        # indices predicted positive by the model
        nonzero = cci_predict.nonzero()[0]
        # get these pairs
        cci_gt_nonzero = cci_gt[nonzero]
        col = cci_gt_nonzero[:, 4]
        row = cci_gt_nonzero[:, 5]

        test_fea = features[-len(self.cell_data) : ]
        labels = self.cell2cluster['cluster'].tolist()
        # import pdb; pdb.set_trace()
        tsne = dict()
        tsne['feature'] = test_fea
        tsne['labels'] = labels
        tsne['src'] = col
        tsne['tar'] = row
        p = self.tsne_path / f'{len(self.cell_data)}.pkl'
        # print(p)
        with open(p, 'wb') as f:
            pkl.dump(tsne, f)


        adj = np.zeros((len(self.cell_data), len(self.cell_data)))
        adj[row, col] = 1
        adj[col, row] = 1

        # import pdb; pdb.set_trace()
        for pair in self.cluster_pairs_enriched:
            type1, type2 = pair
            # print(pair)
            idx1 = self.cluster2cell[type1]
            idx2 = self.cluster2cell[type2]
            total = len(idx1) * len(idx2)
            # import pdb; pdb.set_trace()
            hit = adj[idx1][:, idx2].sum()
            distribution = np.array(self.permuation(adj, idx1, idx2))
            dis = len(np.where(distribution>hit)[0])

            print(f'enriched: hit {dis}, total {num}, ratio {dis/num} ')
        
        for pair in self.cluster_pairs_depleted:
            type1, type2 = pair
            idx1 = self.cluster2cell[type1]
            idx2 = self.cluster2cell[type2]
            total = len(idx1) * len(idx2)
            # import pdb; pdb.set_trace()
            hit = adj[idx1][:, idx2].sum()
            distribution = np.array(self.permuation(adj, idx1, idx2, num))
            dis = len(np.where(distribution>hit)[0]) # larger than original

            print(f'depleted: hit {dis}, total {num}, ratio {dis/num} ')
        
        def one_process(pair, q):
            type1, type2 = pair
            idx1 = self.cluster2cell[type1]
            idx2 = self.cluster2cell[type2]
            total = len(idx1) * len(idx2)
            # import pdb; pdb.set_trace()
            hit = adj[idx1][:, idx2].sum()
            distribution = np.array(self.permuation(adj, idx1, idx2, num))
            dis = len(np.where(distribution>hit)[0]) # larger than original
            q.put(dis)

        wrong = 0
        q = Queue()
        p_obj = list()
        for pair in self.cluster_pairs_unknown:
            p = Process(target=one_process, args=(pair,q,))
            p_obj.append(p)
        for i in p_obj:
            i.start()
        for i in p_obj:
            i.join()
            
        while not q.empty():
            dis = q.get(True)
            if dis / num < 0.05:
                wrong += 1
        print(f'unknown: wrong ratio {wrong} / {len(self.cluster_pairs_unknown)}, {wrong/len(self.cluster_pairs_unknown)} ')
        
    def permuation(self, adj, idx1, idx2, num=10000):
        """permuation the result
        
        Args:
            adj (np.array): shape: (num_cells, num_cells)
            idx1 (np.array): shape: (N1,)
            idx2 (np.array): shape: (N2, )
            num (int, optional): iteration numbers. Defaults to 10000.
        """
        ans = list()
        cells = len(adj)
        # import pdb; pdb.set_trace()
        def swap(cells, idx1, idx2):
            init = list(range(cells))
            for j in list(range(cells))[::-1]:
                idx = random.randint(0,j)
                tmp = init[j]
                init[j] = init[idx]
                init[idx] = tmp
            l1 = list()
            for j in idx1:
                l1.append(init[j])
            l2 = list()
            for j in idx2:
                l2.append(init[j])
            return l1, l2
        
        for i in range(num):
            idx1, idx2 = swap(cells, idx1, idx2)
            hit = adj[idx1][:, idx2].sum()
            ans.append(hit)
        return ans