# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Modelisation of linear relations involved in the optimization problem
"""

import numpy as np
from numpy.polynomial import legendre
from scipy.linalg import block_diag
from scipy.optimize import LinearConstraint


def get_linear_conditions(basis_dimensions, endpoints, coefficients,
                          sigma):
    """
    Return scipy object containing the linear conditions for the optimization
    problem

    Inputs:
        - basis_dimensions: dict
            Give the desired number of basis functions for each variable.
        - endpoints: dict
            Initial and final states that the optimized trajectory must follow.
            ex: {'Var 1': {'start': 109, 'end': 98, 'delta': 10}, ...}
        coefficients: list of ndarray
            Each element of the list is an array containing all the
            coefficients associated with a trajectory
        sigma: ndarray
            Matrix interpreted as an estimated covariance matrix
    
    Outputs:
        - linear_conditions: LinearConstraint object
            Scipy object describing the linear conditions
        - linear_conditions_matrix: ndarray
            Matrix modelling the linear conditions
    """
    # Get matrix and values coming from endpoints conditions
    phi = get_endpoints_matrix(basis_dimensions)
    left_endpoints, right_endpoints = get_endpoints_values(endpoints)
    # Compute intersection between ker sigma and ker phiT phi
    kernel_sigma_phi = get_implicit_matrix(sigma, phi)
    # If intersection is not 0, add implicit conditions
    if kernel_sigma_phi.size > 0:
        # Concatenate endpoints matrix and matrix describing implicit
        # relations
        projected_mean_coefficients = get_implicit_values(kernel_sigma_phi,
                                                        coefficients)
        linear_conditions_matrix = np.concatenate([phi, kernel_sigma_phi.T])
        # Add values from implicit relations to left and right conditions
        list_coefficients = projected_mean_coefficients.tolist()
        linear_conditions_left_values = left_endpoints + list_coefficients
        linear_conditions_right_values = right_endpoints + list_coefficients
    else:
        linear_conditions_matrix = phi
        linear_conditions_left_values = left_endpoints
        linear_conditions_right_values = right_endpoints
    linear_conditions = LinearConstraint(linear_conditions_matrix,
                                         linear_conditions_left_values,
                                         linear_conditions_right_values)

    return linear_conditions, linear_conditions_matrix


def get_implicit_matrix(sigma, phi):
    """
    Compute an orthonormal basis spanning the intersection between
    ker sigma and Im phiT phi - Return a matrix whose columns provide
    the basis and which is involved in a linear condition in the optimization
    problem.
    
    Inputs:
        sigma: ndarray
            Matrix interpreted as an estimated covariance matrix
        Output:
        - phi: ndarray
            Matrix modelling the endpoints conditions

    Output:
        kernel_sigma_phi: ndarray
            Matrix whose columns are orthonormal vectors spanning the
            intersection between ker sigma and Im phiT phi
    """
    # Set arbitrarily a threshold for numerical precision
    threshold = 1e-6
    # Diagonalize sigma
    lambda_sigma, v_sigma = np.linalg.eigh(sigma)
    # Determine the dimension of ker sigma
    sigma_kernel_dim = np.argwhere(lambda_sigma > threshold)[0][0]
    # Compute PhiT phi
    phi_t_phi = np.dot(phi.T, phi)
    # Determine columns of v_sigma which are in ker sigma
    phi_t_phi_v = np.dot(phi_t_phi, v_sigma[:,:sigma_kernel_dim])
    phi_t_phi_v_norms = np.linalg.norm(phi_t_phi_v, axis=0)
    kernel_sigma_phi = np.array([v_sigma[:,k]
                                 for k in range(sigma_kernel_dim)
                                 if phi_t_phi_v_norms[k] < threshold]).T

    return kernel_sigma_phi


def get_implicit_values(kernel_sigma_phi, coefficients):
    """
    Project coefficients mean onto the intersection of ker sigma and
    ker Im PhiT Phi - Provide right-hand side for a linear condition in the
    optimization problem.

    Inputs:
        kernel_sigma_phi: ndarray
            Matrix whose columns are orthonormal vectors spanning the
            intersection between ker sigma and Im phiT phi
        coefficients: list of ndarray
            Each element of the list is an array containing all the
            coefficients associated with a trajectory
    
    Ouput:
        projected_mean_coefficients: ndarray
            Vector given by the product between kernel_sigma_phi.T and
            coefficients mean
    """
    mean_coefficients = np.mean(coefficients, axis=0)
    projected_mean_coefficients = np.dot(kernel_sigma_phi.T, mean_coefficients)

    return projected_mean_coefficients


def get_endpoints_matrix(basis_dimensions):
    """
    Compute a matrix modelling the endpoints conditions involved in the
    optimization problem.

    Inputs:
        - basis_dimensions: dict
            Give the desired number of basis functions for each variable.

    Output:
        - phi: ndarray
            Matrix modelling the endpoints conditions
    """
    # For each variable, compute the values at 0 and T of the d_var Legendre
    # polynomials
    rows_up = []  # Values at 0
    rows_low = []  # Values at T = 1 (because flights defined on [0,1])
    for variable in basis_dimensions:
        row_0 = []
        row_t = []
        for k in range(basis_dimensions[variable]):
            # Create k-th Legendre polynomial on [0,1] and evaluate at
            # the endpoints
            basis_k = legendre.Legendre.basis(k, domain=[0, 1])
            _, basis_k_evaluated = legendre.Legendre.linspace(basis_k,
                                                              n=2)
            row_0.append(basis_k_evaluated[0])
            row_t.append(basis_k_evaluated[1])
        rows_up.append(row_0)
        rows_low.append(row_t)
    # Define upper block
    phi_u = block_diag(*rows_up)
    # Define lower block
    phi_low = block_diag(*rows_low)
    # Concatenate
    phi = np.concatenate((phi_u, phi_low), axis=0)

    return phi


def get_endpoints_values(endpoints):
    """
    Compute right-hand side for a linear condition associated with endpoints
    conditions and involved in the optimization problem

    Input:
        - endpoints: dict
            Initial and final states that the optimized trajectory must follow.
            ex: {'Var 1': {'start': 109, 'end': 98, 'delta': 10}, ...}
    
    Outputs:
        - left_endpoints: list
            List containing the lower conditions (at 0 and T) for each state
        - right_endpoints: list
            List containing the upper conditions (at 0 and T) for each state
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

    return left_endpoints, right_endpoints


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
    phi = get_endpoints_matrix(basis_dimensions)
    linear_endpoints = format_endpoints(phi, endpoints)
    return linear_endpoints, phi


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