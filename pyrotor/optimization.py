# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
File to optimize trajectories
"""

import numpy as np

from cvxopt import solvers
from cvxopt import matrix


def compute_optimized_coefficients(Q, W, phi, lin_const, sigma_inverse,
                                   c_weight, use_quadratic_programming):
    """
    Todo
    """
    if use_quadratic_programming:
        # Use CVXOPT library
        c_opt = minimize_cvx(Q, W, phi, lin_const, sigma_inverse, c_weight)
        c_opt = list(c_opt)

    return c_opt


def minimize_cvx(Q, W, phi, lin_const, sigma_inverse, c_weight):
    """
    Minimize a quadratic function using CVXOPT library - The associated
    matrix must be positive semidefinite

    Inputs:
        Q: ndarray
            Matrix from the quadratic term of the quadratic model
        W: ndarray
            Vector from the linear term of the quadratic model
        phi: ndarray
            Matrix representing the initial and final linear conditions
        lin_const: scipy.optimize.LinearConstraint
            Object containing the initial and final conditions
        sigma_inverse: ndarray
            Pseudoinverse of the covariance matrix of the reference
            coefficients
        c_weight: ndarray
            Coefficients of a weighted trajectory

    Output:
        c_optimized: ndarray
            Coefficients of the optimized trajectory
    """
    # Define quadratic part
    P = matrix(2 * (Q + sigma_inverse))
    # Define linear part
    q = matrix((W - 2 * np.dot(c_weight, sigma_inverse)).T)
    # Define upper and lower endpoints matrix
    # NB: CVXOPT does not deal with two-sided conditions so modify
    # matrix and lower-upper_bounds
    phi = np.concatenate([phi, -phi], axis=0)
    G = matrix(phi, tc='d')
    # Define linear conditions
    lb = -np.array(lin_const.lb)
    h = np.concatenate((lin_const.ub, lb), axis=0)
    h = matrix(h, tc='d')
    # Solve using qp method
    optimize = solvers.qp(P, q, G, h)
    c_optimized = np.array(optimize['x']).ravel()

    return c_optimized