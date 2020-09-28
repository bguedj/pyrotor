# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
File to load a toy dataset
"""

from os.path import dirname, join
from glob import glob
import pandas as pd


def load_toy_dataset(dataset_name):
    """
    Return a dataset with simple trajectories

    Input:
        - dataset_name: str
            Name of the dataset to load
            Among: 'example_1' or 'example_2'

    Output:
        toy_dataset: list of DataFrame
            A list of generic trajectories
    """
    module_path = dirname(__file__)
    dataset_path = 'toy_dataset/' + dataset_name
    toy_dataset_path = join(module_path, dataset_path, '*.csv')
    print(toy_dataset_path)
    toy_dataset_paths = glob(toy_dataset_path)
    toy_dataset = []
    for path in toy_dataset_paths:
        toy_trajectory = pd.read_csv(path, index_col=0)
        toy_dataset.append(toy_trajectory)
    return toy_dataset
