# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Compute the cost of a trajectory or directly on its coefficents
"""

import numpy as np


def compute_f(x, sigma_inverse, c_weight):
    """
    Evaluate the coefficients of a single trajectory over f. Where f is the
    cost function given by the user.

    Inputs:
        - x: ndarray
            Coefficients of a single trajectory.
        - sigma_inverse: ndarray
            Pseudoinverse of the covariance matrix of the reference
            coefficients.
        - c_weight: ndarray
            Coefficients of a weighted trajectory

    Output:
        - cost: float
            The cost of the given trajectory (by its coefficients) over the
            cost function given by the user.
    """
    a = np.dot(np.dot(x.T, sigma_inverse), x)
    truc = np.dot(sigma_inverse, c_weight).T
    b = np.dot(2 * truc, x)
    return a - b


def compute_g(x, Q, W):
    """
    Evaluate the coefficients of a single trajectory over g. Where g is the
    function penalizing the distance between the optimized trajectory and the
    reference trajectories.

    Inputs:
        - x: ndarray
            Coefficients of a single trajectory.
        - Q: ndarray
            Matrix of the quadratic term.
        - W: ndarray
            Vector of the linear term (without intercept).

    Output:
        - g(x): float
            Evaluation of x over g.
    """
    a = np.dot(np.dot(x.T, Q), x)
    b = np.dot(W.T, x)
    return a + b


def compute_cost():
    """
    Compute the cost of a trajectory given the quadratic model of the user.
    """
    pass
