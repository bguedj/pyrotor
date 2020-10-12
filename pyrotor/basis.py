# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Compute means and dot products of functional bases.
"""

import numpy as np
from numpy.polynomial.legendre import Legendre, legmul, legint
from scipy.integrate import quad


def compute_legendre_features(basis_dimension, derivative):
    """
    Format means and dot products of Legendre polynomials (and their
    derivatives) for the optimization problem.

    Inputs:
        - basis_dimension: dict
            Give the number of basis functions for each state
        - derivative: boolean
            Take into account or not derivatives of states
    Outputs:
        - mean: ndarray
            Array containing the means over an interval
        - dot_product: ndarray
            Matrix containing the dot products
    """
    # Compute means and dot products only for states
    mean, dot_product = compute_legendre_mean_dot_product(basis_dimension)
    if derivative:
        # Compute means for derivatives of states
        mean_deriv = compute_legendre_mean_derivatives(basis_dimension)
        # Compute dot products for states and derivatives of states
        dot_product_12, dot_product_22 = \
            compute_legendre_dot_product_derivatives(basis_dimension)
        # Concatenate vectors and matrices to format
        mean = np.concatenate((mean, mean_deriv))
        dot_product_1 = np.concatenate((dot_product, dot_product_12), axis=1)
        dot_product_2 = np.concatenate((dot_product_12.T, dot_product_22),
                                       axis=1)
        dot_product = np.concatenate((dot_product_1, dot_product_2), axis=0)

    return mean, dot_product


def compute_legendre_mean_dot_product(basis_dimension):
    """
    Compute means and dot products of Legendre polynomials.

    Input:
        - basis_dimension: dict
            Give the number of basis functions for each state
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
    for state in basis_dimension:
        mean[i] += duration
        i += basis_dimension[state]
        
    # Compute dot product between the polynomials
    # Here use <P_i, P_j> = duration / (2*i + 1) * delta_il=j
    dot_product = np.zeros([dimension, dimension])
    i, j = 0, 0
    for state1 in basis_dimension:
        for state2 in basis_dimension:
            k_range = min([basis_dimension[state1],
                           basis_dimension[state2]])
            for k in range(k_range):
                # Squared L^2-norm of the k-th Legendre polynomial
                dot_product[i + k, j + k] = duration / (2*k + 1)
            j += basis_dimension[state2]
        j = 0
        i += basis_dimension[state1]

    return mean, dot_product


def compute_legendre_mean_derivatives(basis_dimension):
    """
    Compute means of Legendre polynomials derivatives.

    Input:
        - basis_dimension: dict
            Give the number of basis functions for each state
    Output:
        - mean_deriv: ndarray
            Array containing the means over an interval
    """
    mean_deriv = np.array([])
    for state in basis_dimension:
        # Use Legendre property: P(1) = 1, P(0)= 1 or -1
        x = [k % 2 * 2 for k in range(basis_dimension[state])]
        mean_deriv = np.append(mean_deriv, x)

    return mean_deriv


def compute_legendre_dot_product_derivatives(basis_dimension):
    """
    Compute dot products of Legendre polynomials and their derivatives.

    Input:
        - basis_dimension: dict
            Give the number of basis functions for each state
    Outputs:
        - dot_product_12: ndarray
            Array containing the dot products of Legendre polynomials
            with their derivatives
        - dot_product_22: ndarray
            Array containing the dot products of Legendre polynomials
            derivatives
    """
    # Compute the dimension of the problem
    dimension = np.sum([basis_dimension[elt] for elt in basis_dimension])
    dot_product_12 = np.zeros([dimension, dimension])
    dot_product_22 = np.zeros([dimension, dimension])
    i, j = 0, 0
    # Loop over states
    for state1 in basis_dimension:
        for state2 in basis_dimension:
            for k in range(basis_dimension[state1]):
                c_k = np.zeros(basis_dimension[state1])
                c_k[k] += 1
                # Create Legendre class for k-th polynomial
                c_k = Legendre(c_k, domain=[0, 1])
                # Compute derivative
                c_k_deriv = c_k.deriv()
                for l in range(basis_dimension[state2]):
                    c_l = np.zeros(basis_dimension[state2])
                    c_l[l] += 1
                    # Create Legendre class for k-th polynomial
                    c_l = Legendre(c_l, domain=[0, 1])
                    # Compute derivative
                    c_l_deriv = c_l.deriv()
                    # Multiply polynomials
                    product_12 = legmul(list(c_k), list(c_l_deriv))
                    product_22 = legmul(list(c_k_deriv), list(c_l_deriv))
                    # Create classes
                    product_12 = Legendre(product_12, domain=[0, 1])
                    product_22 = Legendre(product_22, domain=[0, 1])
                    # Integrate
                    int_product_12 = product_12.integ()
                    int_product_22 = product_22.integ()
                    # Evaluate at the endpoints
                    _, traj_deriv_12 = int_product_12.linspace(n=2)
                    _, traj_deriv_22 = int_product_22.linspace(n=2)
                    # Deduce dot products
                    dot_product_12[i + k, j + l] += traj_deriv_12[1]
                    dot_product_12[i + k, j + l] -= traj_deriv_12[0]
                    dot_product_22[i + k, j + l] += traj_deriv_22[1]
                    dot_product_22[i + k, j + l] -= traj_deriv_22[0]
            j += basis_dimension[state2]
        j = 0
        i += basis_dimension[state1]

    return dot_product_12, dot_product_22
