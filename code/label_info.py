import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def get_label_info(params):
    tissue = params.tissue
    train = params.train
    test = params.test
    print(type(train), type(test))
    proj_path = Path(__file__).parent.resolve().parent.resolve()
    mouse_data_path = proj_path / 'data' / 'mouse_data'

    train_set, test_set = set(), set()
    train_files = [mouse_data_path / f'mouse_{tissue}{i}_celltype.csv' for i in train]
    test_files = [mouse_data_path / f'mouse_{tissue}{i}_celltype.csv' for i in test]
    for file in train_files:
        train_set = set(pd.read_csv(file, dtype=np.str, header=0).values[:, 2]) | train_set
    for file in test_files:
        test_set = set(pd.read_csv(file, dtype=np.str, header=0).values[:, 2]) | test_set

    print(f'{len(train_set)} labels in train_set, {len(test_set)} labels in test_set.')
    print(test_set <= train_set)


if __name__ == '__main__':
    'python ./code/label_info.py --tissue Peripheral_blood --train 2466 --test 135 283 352 658 3201'
    parser = argparse.ArgumentParser()
    parser.add_argument('--tissue', required=True, type=str)
    parser.add_argument('--train', required=True, nargs='+')
    parser.add_argument('--test', required=True, nargs='+')
    params = parser.parse_args()

    get_label_info(params)
