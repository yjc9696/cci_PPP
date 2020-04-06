#!/usr/local/bin/python
# -*- coding: utf-8 -*-
"""
to construct the cci throgh the result
"""
import numpy as np
import pandas as pd

class Evaluate:
    def __init__(self, params):
        """init the class
        
        Args:
            params (args): [description]
        """
        proj_path = Path(__file__).parent.resolve().parent.resolve()
        mouse_data_path = proj_path / 'data' / params.data_dir #small_intestine
        # train_dataset = mouse_data_path / params.train_dataset
        test_dataset = mouse_data_path / params.test_dataset

        cell2cluter_path = test_dataset / params.cell_cluster
        cell_data_path = test_dataset / params.cell_data_path
        # ligand_receptor_path = mouse_data_path / params.ligand_receptor_gene
        cluster2cluster_enriched_path = mouse_data_path / params.cluster_cluster_interaction_enriched
        cluster2cluster_depleted_path = mouse_data_path / params.cluster_cluster_interaction_depleted

        self.cell_data = pd.read_csv(cell_data_path, index_col=0).fillna(0)
        self.cell2cluster = pd.read_csv(cell2cluster_path, index_col=0)
        assert self.cell2cluster.index.tolist() == self.cell_data.columns.tolist(), 'cell data and cell_cluster not match, error'
        
        # 'cluster1', 'cluster2'
        self.cluster2cluster_enriched = pd.read_csv(cluster2cluster_enriched_path, header=0, index_col=0)
        self.cluster2cluster_depleted = pd.read_csv(cluster2cluster_depleted_path, header=0, index_col=0)