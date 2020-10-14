# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Test the optimization module
"""
import pytest

import numpy as np

from cvxopt import solvers
from cvxopt import matrix

from scipy.optimize import LinearConstraint

from pyrotor.optimization import minimize_cvx
from pyrotor.optimization import minimize_trust_constr


def test_minimize_cvx():
    # Solve min ||x-np.ones||_2^2 such that x1 + x2 = 2 and x_{K-1} - x_K = 0
    # Here P = Id, q = -2 * np.ones, phi = [[1 1 0 ... 0 0],
    #                                       [0 0 0 ... 1 -1]],
    # sigma_inverse and c_weight are randomly chosen
    basis_dimension = {"A": 3, "B": 2}
    # Dimension of the problem
    K = np.sum([basis_dimension[elt] for elt in basis_dimension])
    # Create a random vector
    c_weight = np.random.rand(K)
    # Create a random matrix
    sigma_inverse = np.random.rand(K, K)
    # Write Q = Id - sigma_inverse so that P = 2/ 2 * (Q + sigma_inverse) = Id
    Q = np.eye(K) - sigma_inverse
    # Write W = -2 * np.ones + 2 * c_weight * sigma_inverse so that
    # q = W - 2 * c_weight * sigma_inverse = -2 * np.ones
    W = -2 * np.ones(K) + 2 * np.dot(sigma_inverse, c_weight)
    # Create phi
    phi = np.zeros([2, K])
    phi[0, :2] += 1
    phi[1, -1] += -1
    phi[1, -2] += 1
    # Create b for linear conditions phi * c = b
    left_hs = [1.9, -.1]
    right_hs = [2.1, .1]
    # Create LinearConstraint object
    lin_const = LinearConstraint(phi, left_hs, right_hs)
    # Apply minimize_cvx()
    kappa = 1
    c_optimized = minimize_cvx(Q, W, phi, lin_const, sigma_inverse,
                               c_weight, kappa)

    expected_c_optimized = np.ones(K)

    np.testing.assert_almost_equal(c_optimized, expected_c_optimized)

def test_minimize_trust_constr():
    # Solve min ||x-np.ones||_2^2 such that x1 + x2 = 2 and x_{K-1} - x_K = 0
    # Here P = Id, q = -2 * np.ones, phi = [[1 1 0 ... 0 0],
    #                                       [0 0 0 ... 1 -1]],
    # sigma_inverse and c_weight are randomly chosen
    basis_dimension = {"A": 3, "B": 2}
    # Dimension of the problem
    K = np.sum([basis_dimension[elt] for elt in basis_dimension])
    # Create a random vector
    c_weight = np.random.rand(K)
    # Create a random matrix
    sigma_inverse = np.random.rand(K, K)
    # Write Q = Id - sigma_inverse so that P = 2/ 2 * (Q + sigma_inverse) = Id
    Q = np.eye(K) - sigma_inverse
    # Write W = -2 * np.ones + 2 * c_weight * sigma_inverse so that
    # q = W - 2 * c_weight * sigma_inverse = -2 * np.ones
    W = -2 * np.ones(K) + 2 * np.dot(sigma_inverse, c_weight)
    # Create phi
    phi = np.zeros([2, K])
    phi[0, :2] += 1
    phi[1, -1] += -1
    phi[1, -2] += 1
    # Create b for linear conditions phi * c = b
    left_hs = [1.9, -.1]
    right_hs = [2.1, .1]
    # Create LinearConstraint object
    lin_const = LinearConstraint(phi, left_hs, right_hs)
    # Apply minimize_cvx()
    kappa = 1
    c_optimized = minimize_trust_constr(Q, W, phi, lin_const, sigma_inverse, c_weight, kappa)

    expected_c_optimized = np.ones(K)

    np.testing.assert_almost_equal(c_optimized, expected_c_optimized, decimal=6)
