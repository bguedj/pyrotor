# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Test the objective_matrices module
"""
import unittest

import numpy as np
from numpy.polynomial import legendre

from pyrotor.objective_matrices import extend_matrix
from pyrotor.objective_matrices import compute_objective_matrices


# TODO: test for model_to_matrix (train a model before)


def test_extend_matrix():
    w = np.array([1., 2.])
    q = np.array([[1., 2.],
                  [2., 4.]])
    mean = np.array([1., 2., 3., 1., 2.])
    dot_product = np.array([[1., 0., 0., 1., 0.],
                            [0., 2., 0., 0., 2.],
                            [0., 0., 3., 0., 0.],
                            [1., 0., 0., 1., 0.],
                            [0., 2., 0., 0., 2.]])
    basis_dimension = {"A": 3, "B": 2}
    W, Q = extend_matrix(w, q, mean, dot_product, basis_dimension)
    expected_W = np.array([1., 2., 3., 2., 4.])
    expected_Q = np.array([[1., 0., 0., 2., 0.],
                           [0., 2., 0., 0., 4.],
                           [0., 0., 3., 0., 0.],
                           [2., 0., 0., 4., 0.],
                           [0., 4., 0., 0., 8.]])

    np.testing.assert_almost_equal(W, expected_W)
    np.testing.assert_almost_equal(Q, expected_Q)


def test_compute_objective_matrices():
    # TODO: test in case 'quad_model' is a path for a trained model
    basis = 'legendre'
    basis_dimension = {"A": 3, "B": 2}
    c = 0
    w = np.array([1., 2.])
    q = np.array([[1., 2.],
                  [2., 4.]])
    quad_model = [c, w, q]
    W, Q = compute_objective_matrices(basis, basis_dimension, quad_model)
    expected_W = np.array([1., 0., 0., 2., 0.])
    expected_Q = np.array([[1., 0., 0., 2., 0.],
                           [0., 1/3, 0., 0., 2/3],
                           [0., 0., 1/5, 0., 0.],
                           [2., 0., 0., 4., 0.],
                           [0., 2/3, 0., 0., 4/3]])

    np.testing.assert_almost_equal(W, expected_W)
    np.testing.assert_almost_equal(Q, expected_Q)
