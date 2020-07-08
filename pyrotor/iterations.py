# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Module used to iterate
"""

import numpy as np
from copy import copy


def binary_search_best_trajectory(optimization, i, step):
    """
    FIXME: avoid using both kappa and weight dict.
    """
    optimization.i_kappa = i
    optimization.i_binary_search += 1
    print("Kappa #"+str(i))
    if i < 0:
        message = "Trajectories of reference too close to constraints"
        raise ValueError(message)
    optimization.weight_dict = {k: optimization.kappas[i]
                                * optimization.original_weights[k]
                                for k in optimization.original_weights}
    optimization.compute_trajectory()

    step = step//2
    if not optimization.is_valid:
        if step == 0:
            step = 1
        binary_search_best_trajectory(optimization, i-step, step)
    else:
        if len(optimization.kappas)-1 != i and step != 0:
            binary_search_best_trajectory(optimization, i+step, step)


def iterate(optimization, iteration_function):
    """
    FIXME: avoid using both kappa and weight dict.
    """
    optimization.original_weights = copy(optimization.weight_dict)
    x = np.linspace(0, 1, 30)
    optimization.kappas = 1/np.exp(5*x**(1/2))
    optimization.i_binary_search = 0
    optimization.binary_search_best_trajectory(len(optimization.kappas)-1,
                                               len(optimization.kappas)-1)
    if not optimization.is_valid:
        message = "Trajectories of reference too close to constraints"
        raise ValueError(message)
