# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Modelisation of linear relations
"""

import numpy as np
from numpy.polynomial import legendre
from scipy.linalg import block_diag
from scipy.optimize import LinearConstraint

# rename file
# TODO: add conditions from 'natural' correlations (V_2 matrix)


def get_linear_conditions():
    pass


def get_implicit_matrix(sigma_inverse, phi):
    """
    """
    pass


def get_linear_endpoints(basis_dimensions, endpoints):
    """
    Return the linear endpoints that must be satified at the initial and final
    states.

    Inputs:
        - basis_dimensions: dict
            Defining the dimension for each variable. ex: {"var 1": 5, ...}
        - vector_omega: ndarray
            Weights to apply on the reference trajectories when computing the
            endpoints.
        - ref_trajectories: list of Pandas DataFrame
            Trajectories of reference.
        - endpoints: dict
            Initial and final states that the optimized trajectory must follow.
            ex: {'Var 1': {'start': 109, 'end': 98, 'delta': 10}, ...}
    """
    phi = get_endpoints_matrix(basis_dimensions, endpoints)
    linear_endpoints = format_endpoints(phi, endpoints)
    return linear_endpoints, phi


def get_endpoints_matrix(basis_dimensions, endpoints):
    """
    Compute a matrix modelling the endpoints used for our optimization problem.

    Inputs:
        - basis_dimensions: dict
            Give the desired number of basis functions for each variable.

    Output:
        - phi: numpy array [2 * n_var, d]
            Matrix modelling the endpoints endpoints
    """
    phi_width = sum(basis_dimensions.values())
    phi_height = 2 * len(endpoints)
    phi = np.zeros((phi_height, phi_width))
    i = j = 0
    for variable in basis_dimensions:
        if variable in endpoints:
            for k in range(basis_dimensions[variable]):
                basis_k = legendre.Legendre.basis(k, domain=[0, 1])
                _, basis_k_evaluated = legendre.Legendre.linspace(basis_k, n=2)
                phi[j, i] = basis_k_evaluated[0]
                phi[phi_height//2+j, i] = basis_k_evaluated[1]
                i += 1
            j += 1
        else:
            i += basis_dimensions[variable]
    return phi


def get_endpoints_values(endpoints):
    pass


def format_endpoints(phi, endpoints):
    """
    Build a Scipy object modelling endpoints and used for optimization

    Inputs:
        - phi: ndarray
            Matrix modelling the endpoints
        - endpoints: dict
            Initial and final states that the optimized trajectory must follow.
            ex: {'Var 1': {'start': 109, 'end': 98, 'delta': 10}, ...}
    Output:
        - linear_endpoints: LinearConstraint object
            Scipy object describing the linear endpoints
    """
    # Define lower endpoints at 0
    left_endpoints_0 = [endpoints[variable]['start']
                        - endpoints[variable]['delta']
                        for variable in endpoints]
    # Define lower endpoints at T
    left_endpoints_t = [endpoints[variable]['end']
                        - endpoints[variable]['delta']
                        for variable in endpoints]
    # Merge to obtain lower endpoints
    left_endpoints = left_endpoints_0 + left_endpoints_t
    # Define upper endpoints at 0
    right_endpoints_0 = [endpoints[variable]['start']
                         + endpoints[variable]['delta']
                         for variable in endpoints]
    # Define upper endpoints at T
    right_endpoints_t = [endpoints[variable]['end']
                         + endpoints[variable]['delta']
                         for variable in endpoints]
    # Merge to obtain upper endpoints
    right_endpoints = right_endpoints_0 + right_endpoints_t

    linear_endpoints = LinearConstraint(phi, left_endpoints, right_endpoints)

    return linear_endpoints


def get_implicit_values():
    """

    """
    pass
