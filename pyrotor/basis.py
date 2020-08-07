# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Compute means and dot products of a given functional basis
"""

import numpy as np


def compute_legendre_features(basis_dimension):
    """
    Compute means and dot products of the Legendre polynomial basis - Return
    vector and matrix in the format adapted to the optimization problem

    Inputs:
        - basis_dimension: dict
            Give the number of basis functions for each variable

    Outputs:
        - mean: ndarray
            Array containing the means over an interval
        - dot_product: ndarray
            Matrix containing the dot products
    """
    # Trajectories are formally defined on [0,1] so the interval length is 1
    duration = 1
    # Compute the dimension of the problem
    dimension = np.sum([basis_dimension[elt] for elt in basis_dimension])
    # For Legendre polynomials, mean = 0 except when k = 0
    mean = np.zeros(dimension)
    i = 0
    for variable in basis_dimension:
        mean[i] += duration
        i += basis_dimension[variable]
    # Compute dot product between the polynomials
    # Here use <P_i, P_j> = duration / (2*i + 1) * delta_il=j
    dot_product = np.zeros([dimension, dimension])
    i, j = 0, 0
    for variable1 in basis_dimension:
        for variable2 in basis_dimension:
            k_range = min([basis_dimension[variable1],
                           basis_dimension[variable2]])
            for k in range(k_range):
                # Squared L^2-norm of the k-th Legendre polynomial
                dot_product[i + k, j + k] = duration / (2*k + 1)
            j += basis_dimension[variable2]
        j = 0
        i += basis_dimension[variable1]

    return mean, dot_product
