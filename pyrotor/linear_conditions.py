# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Modelisation of linear relations involved in the optimization problem
"""

import numpy as np
from numpy.polynomial import legendre
from numpy.linalg import multi_dot
from scipy.linalg import block_diag, null_space
from scipy.optimize import LinearConstraint


def get_linear_conditions(basis, basis_dimensions, endpoints, coefficients,
                          sigma):
    """
    Return scipy object containing the linear conditions for the
    optimization problem.

    Inputs:
        - basis: string
            Name of the functional basis
        - basis_dimensions: dict
            Give the desired number of basis functions for each state
        - endpoints: dict
            Initial and final states that the optimized trajectory must
            satisfy
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
    # Get matrix and bounds coming from endpoints conditions
    phi = get_endpoints_matrix(basis, basis_dimensions, endpoints)
    states = basis_dimensions.keys()
    left_endpoints, right_endpoints = get_endpoints_bounds(endpoints, states)
    # Compute intersection between null space of sigma and phiT phi
    null_space_sigma_phi = get_implicit_matrix(sigma, phi)

    # If intersection is not 0, add implicit conditions
    if null_space_sigma_phi.size > 0:
        # Concatenate endpoints matrix and matrix describing implicit
        # relations
        linear_conditions_matrix = np.concatenate([phi,
                                                  null_space_sigma_phi.T])
        # Add bounds from implicit relations to left and right conditions
        projected_mean_coefficients = get_implicit_bounds(
                                        null_space_sigma_phi,
                                        coefficients)
        list_coefficients = projected_mean_coefficients.tolist()
        linear_conditions_left_bounds = left_endpoints + list_coefficients
        linear_conditions_right_bounds = right_endpoints + list_coefficients
    else:
        linear_conditions_matrix = phi
        linear_conditions_left_bounds = left_endpoints
        linear_conditions_right_bounds = right_endpoints

    linear_conditions = LinearConstraint(linear_conditions_matrix,
                                         linear_conditions_left_bounds,
                                         linear_conditions_right_bounds)

    return linear_conditions, linear_conditions_matrix


def get_implicit_matrix(sigma, phi):
    """
    Compute an orthonormal basis spanning the intersection between null
    space of sigma and phiT * phi. Return a matrix whose columns
    provide the basis and which is involved in implicit conditions in
    the optimization problem.

    Inputs:
        - sigma: ndarray
            Matrix interpreted as an estimated covariance matrix
        - phi: ndarray
            Matrix modelling the endpoints conditions
    Output:
        - null_space_sigma_phi: ndarray
            Matrix whose columns are orthonormal vectors and spanning
            the intersection between null space of sigma and phiT * phi
    """
    # Set arbitrarily a threshold for numerical precision
    threshold = 1e-10
    # Diagonalize sigma
    lambda_sigma, v_sigma = np.linalg.eigh(sigma)
    # Determine the dimension of null space of sigma and keep eigenvectors
    # associated with null eigenvalues
    sigma_null_space_dim = np.argwhere(lambda_sigma > threshold)[0][0]
    v_sigma_null_space = v_sigma[:, :sigma_null_space_dim]
    # If no linear relations detected, return empty matrix
    if v_sigma_null_space.size == 0:
        return np.array([])

    # Compute phiT * phi * v_sigma_null_space to find vectors in
    # v_sigma_null_space which are also in null space of phiT * phi
    product = multi_dot([phi.T, phi, v_sigma_null_space])
    # Determine a combination of columns of v_sigma_null_space which are in
    # null space of phiT * phi
    null_space_product = null_space(product)
    null_space_sigma_phi = np.dot(v_sigma_null_space, null_space_product)

    return null_space_sigma_phi


def get_implicit_bounds(null_space_sigma_phi, coefficients):
    """
    Project coefficients mean onto the intersection of null space of
    sigma and Im PhiT * Phi.

    Provide right-hand side for a linear condition in the optimization
    problem.

    Inputs:
        null_space_sigma_phi: ndarray
            Matrix whose columns are orthonormal vectors spanning the
            intersection between null space of sigma and phiT * phi
        coefficients: list of ndarray
            Each element of the list is an array containing all the
            coefficients associated with a trajectory
    Ouput:
        projected_mean_coefficients: ndarray
            Vector given by the product between null_space_sigma_phi.T
            and coefficients mean
    """
    mean_coefficients = np.mean(coefficients, axis=0)
    projected_mean_coefficients = np.dot(null_space_sigma_phi.T,
                                         mean_coefficients)

    return projected_mean_coefficients


def get_endpoints_matrix(basis, basis_dimensions, endpoints):
    """
    Compute a matrix modelling the endpoints conditions involved in the
    optimization problem.

    The upper-block of the matrix contains the starting conditions
    while the lower-blocks contains the ending conditions.

    Inputs:
        - basis: string
            Name of the functional basis
        - basis_dimensions: dict
            Give the desired number of basis functions for each state
        - endpoints: dict
            Initial and final states that the optimized trajectory must
            satisfy
            ex: {'Var 1': {'start': 109, 'end': 98, 'delta': 10}, ...}
    Output:
        - phi: ndarray
            Matrix modelling the endpoints conditions
    """
    # Define matrix shape
    phi_width = sum(basis_dimensions.values())
    phi_height = 2 * len(endpoints)
    phi = np.zeros((phi_height, phi_width))
    i = j = 0
    # Legendre case - Fill using values of Legendre polynomials at the
    # endpoints
    if basis == 'legendre':
        for state in basis_dimensions:
            if state in endpoints:
                for k in range(basis_dimensions[state]):
                    basis_k = legendre.Legendre.basis(k, domain=[0, 1])
                    _, basis_k_evaluated = legendre.Legendre.linspace(basis_k,
                                                                      n=2)
                    # Define upper-block
                    phi[i, j] = basis_k_evaluated[0]
                    # Define lower-block
                    phi[phi_height//2+i, j] = basis_k_evaluated[1]
                    j += 1
                i += 1
            else:
                j += basis_dimensions[state]
    # B-spline case - Fill using the fact that only the first and last splines
    # are non-zero at the endpoints with value 1
    elif basis == 'bspline':
        # Loop over states
        for state in basis_dimensions:
            if state in endpoints:
                phi[i, j] = 1.
                phi[phi_height//2+i, j+basis_dimensions[state]-1] = 1.
                i += 1
            j += basis_dimensions[state]

    return phi


def get_endpoints_bounds(endpoints, states):
    """
    Compute right-hand side for a linear condition associated with
    endpoints conditions and involved in the optimization problem.

    Input:
        - endpoints: dict
            Initial and final states that the optimized trajectory must
            satisfy
            ex: {'Var 1': {'start': 109, 'end': 98, 'delta': 10}, ...}
        - states: iterable
            State names
    Outputs:
        - left_endpoints: list
            List containing the lower conditions (at 0 and T) for each
            state
        - right_endpoints: list
            List containing the upper conditions (at 0 and T) for each
            state
    """
    # Define lower endpoints at 0
    left_endpoints_0 = [endpoints[state]['start']
                        - endpoints[state]['delta']
                        for state in states
                        if state in endpoints.keys()]
    # Define lower endpoints at T
    left_endpoints_t = [endpoints[state]['end']
                        - endpoints[state]['delta']
                        for state in states
                        if state in endpoints.keys()]
    # Merge to obtain lower endpoints
    left_endpoints = left_endpoints_0 + left_endpoints_t

    # Define upper endpoints at 0
    right_endpoints_0 = [endpoints[state]['start']
                         + endpoints[state]['delta']
                         for state in states
                         if state in endpoints.keys()]
    # Define upper endpoints at T
    right_endpoints_t = [endpoints[state]['end']
                         + endpoints[state]['delta']
                         for state in states
                         if state in endpoints.keys()]

    # Merge to obtain upper endpoints
    right_endpoints = right_endpoints_0 + right_endpoints_t

    return left_endpoints, right_endpoints
