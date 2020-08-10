# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
File to load a toy dataset.
"""

from os.path import dirname, join
from glob import glob
import pandas as pd


def load_toy_dataset():
    module_path = dirname(__file__)
    toy_dataset_path = join(module_path, 'toy_dataset', '*.csv')
    print(toy_dataset_path)
    toy_dataset_paths = glob(toy_dataset_path)
    toy_dataset = []
    for path in toy_dataset_paths:
        toy_trajectory = pd.read_csv(path, index_col=0)
        toy_dataset.append(toy_trajectory)
    return toy_dataset
