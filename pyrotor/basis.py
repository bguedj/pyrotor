# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Compute means and dot products of a given functional basis
"""

import numpy as np


def basis_legendre(basis_dimension):
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
    T = 1
    # Compute the dimension of the problem
    d = np.sum([basis_dimension[elt] for elt in basis_dimension])
    # For Legendre polynomials, mean = 0 except when k = 0
    mean = np.zeros(d)
    k = 0
    for variable in basis_dimension:
        mean[k] += T
        k += basis_dimension[variable]
    # Compute dot product between the polynomials
    # Here use <P_k, P_l> = T / (2*k + 1) * delta_kl
    dot_product = np.zeros([d, d])
    k, l = 0, 0
    for variable1 in basis_dimension:
        for variable2 in basis_dimension:
            m = min([basis_dimension[variable1],
                     basis_dimension[variable2]])
            for n in range(m):
                # Squared L^2-norm of the n-th Legendre polynomial
                dot_product[k + n, l + n] = T / (2*n + 1)
            l += basis_dimension[variable2]
        l = 0
        k += basis_dimension[variable1]

    return mean, dot_product