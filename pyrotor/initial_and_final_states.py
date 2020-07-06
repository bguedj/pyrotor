# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Modelisation of optimization endpoints
"""

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import LinearConstraint
from numpy.polynomial import legendre


def get_linear_endpoints(basis_dimensions, vector_omega,
                         ref_trajectories, endpoints_delta):
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
        - endpoints_delta: dict
            Defining the maximum error at the initial and final states for each
            variable. ex: {"var 1": 10, ...}
    """
    Phi = build_matrix_endpoints(basis_dimensions)
    endpoints = {var: {'delta': endpoints_delta[var]}
                 for var in endpoints_delta}
    endpoints = compute_weighted_average_endpoints(endpoints,
                                                   vector_omega,
                                                   ref_trajectories)
    linear_endpoints = format_endpoints(Phi, endpoints)
    return linear_endpoints


def build_matrix_endpoints(basis_dimensions):
    """
    Compute a matrix modelling the endpoints used for our optimization problem.

    Inputs:
        - basis_dimensions: dict
            Give the desired number of basis functions for each variable.

    Output:
        - phi: numpy array [2 * n_var, d]
            Matrix modelling the endpoints endpoints
    """
    # For each variable, compute the values at 0 and T of the d_var Legendre
    # polynomials
    rows_up = []  # Values at 0
    rows_low = []  # Values at T = 1 (because flights defined on [0,1])
    for variable in basis_dimensions:
        row_0 = []
        row_T = []
        for k in range(basis_dimensions[variable]):
            # Create k-th Legendre polynomial on [0,1] and evaluate at
            # the endpoints
            basis_k = legendre.Legendre.basis(k, domain=[0, 1])
            _, basis_k_evaluated = legendre.Legendre.linspace(basis_k,
                                                              n=2)
            row_0.append(basis_k_evaluated[0])
            row_T.append(basis_k_evaluated[1])
        rows_up.append(row_0)
        rows_low.append(row_T)
    # Define upper block
    phi_u = block_diag(*rows_up)
    # Define lower block
    phi_low = block_diag(*rows_low)
    # Concatenate
    phi = np.concatenate((phi_u, phi_low), axis=0)

    return phi


def compute_weighted_average_endpoints(endpoints, omega, ref_trajectories):
    """
    For each variable, it computes the weighted average of the ref trajectories
    initial and final states.

    Inputs:
        - endpoints: dict
            Contain endpoints error margins for all the variables
            ex: {"var 1": {"delta": 10}}
        - omega: ndarray
            Vector containing the weight for each reference flight
        - ref_trajectories: list
            List of DataFrame, each DataFrame describing a reference flight

    Ouput:
        - endpoints: dict
            Initial and final states that the optimized trajectory must follow.
            ex: {'Var 1': {'start': 109, 'end': 98, 'delta': 10}, ...}
    """
    for variable in ref_trajectories[0].columns:
        endpoints[variable]['start'] = np.average(
            [y_ref_i[variable].values[0] for y_ref_i in ref_trajectories],
            weights=omega)
        endpoints[variable]['end'] = np.average(
            [y_ref_i[variable].values[-1] for y_ref_i in ref_trajectories],
            weights=omega)
    return endpoints


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
    left_endpoints_T = [endpoints[variable]['end'] - endpoints[variable]['delta']
                        for variable in endpoints]
    # Merge to obtain lower endpoints
    left_endpoints = left_endpoints_0 + left_endpoints_T
    # Define upper endpoints at 0
    right_endpoints_0 = [endpoints[variable]['start'] + endpoints[variable]['delta']
                         for variable in endpoints]
    # Define upper endpoints at T
    right_endpoints_T = [endpoints[variable]['end'] + endpoints[variable]['delta']
                         for variable in endpoints]
    # Merge to obtain upper endpoints
    right_endpoints = right_endpoints_0 + right_endpoints_T

    linear_endpoints = LinearConstraint(phi, left_endpoints, right_endpoints)

    return linear_endpoints
