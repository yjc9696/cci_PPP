import argparse

import os
import pandas as pd
import torch
from scipy.sparse import csr_matrix, vstack
from sklearn.decomposition import PCA
from pathlib import Path
import numpy as np
import pickle as pkl


class ChooseGene(object):
    def __init__(self, tissue, train):
        self.proj_path = Path('.')
        self.mouse_data_path = self.proj_path / 'data' / 'mouse_data'
        self.statistics = self.mouse_data_path / 'statistics'
        
        self.train_files = self.mouse_data_path.glob(f'*{tissue}*_data.csv')
        self.genes = dict()

    def process_one_file(self, file):

        data = pd.read_csv(file, sep=',', dtype=np.str, header=0, index_col=0)
        data = data.fillna(0)
        data = data.applymap(lambda x: float(x))
        data['sum'] = data.apply(np.sum, axis=1)
        for index, num in zip(data.index, data['sum']):
            if index in self.genes.keys():
                self.genes[index] += num
            else:
                self.genes[index] = num

    def process(self):
        for file in self.train_files:
            print(file)
            self.process_one_file(file)
        
        
    def choose_gene(self, rate=0.5, load=False):
        if load:
            with open(self.statistics / 'gene.pkl', 'rb') as f:
                self.genes = pkl.load(f)
        else:
            self.process()
            with open(self.statistics / 'gene.pkl', 'wb') as f:
                pkl.dump(self.genes, f)

        print(f'gene total number is {len(self.genes)}')
        ave = sum(self.genes.values()) / len(self.genes) * rate

        with open(self.statistics / 'Mammary_gland_genes.txt', 'w', encoding='utf-8') as f:
            for key, val in self.genes.items():
                if val > ave:
                    f.write(key+'\n')

if __name__ == '__main__':
    """
    python ./code/datasets/choose_genes.py --train 3510 1311 6633 6905 4909 2081 --tissue Mammary_gland
    """
    print('change work dir')
    os.chdir('/home/yangyueren/code/bio_ai')
    parser = argparse.ArgumentParser()
    parser.add_argument('--tissue', required=True, type=str)
    parser.add_argument('--train', required=True, nargs='+')
    parser.add_argument('--rate', required=True, type=float)
    params = parser.parse_args()
    
    train = ['3510', '1311', '6633', '6905', '4909', '2081']
    tissue = 'Mammary_gland'
    gene = ChooseGene(tissue, train)
    # gene.process()
    gene.choose_gene(rate=params.rate, load=False)
