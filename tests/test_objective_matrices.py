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
    basis_dimension = {"A": 3, "B": 2}
    W, Q = extend_matrix(w, q, basis_dimension, derivative=False)
    expected_W = np.array([1., 1., 1., 2., 2.])
    expected_Q = np.array([[1., 1., 1., 2., 2.],
                           [1., 1., 1., 2., 2.],
                           [1., 1., 1., 2., 2.],
                           [2., 2., 2., 4., 4.],
                           [2., 2., 2., 4., 4.]])

    np.testing.assert_almost_equal(W, expected_W)
    np.testing.assert_almost_equal(Q, expected_Q)

    w = np.array([1., 2., 0., 3.])
    q = np.array([[1., 2., 3., 4.],
                  [2., 4., 6., 8.],
                  [3., 6., 8., 4.],
                  [4., 8., 4., 16.]])
    W, Q = extend_matrix(w, q, basis_dimension, derivative=True)
    expected_W = np.array([1., 1., 1., 2., 2., 0., 0., 0., 3., 3.])
    expected_Q = np.array([[1., 1., 1., 2., 2., 3., 3., 3., 4., 4.],
                           [1., 1., 1., 2., 2., 3., 3., 3., 4., 4.],
                           [1., 1., 1., 2., 2., 3., 3., 3., 4., 4.],
                           [2., 2., 2., 4., 4., 6., 6., 6., 8., 8.],
                           [2., 2., 2., 4., 4., 6., 6., 6., 8., 8.],
                           [3., 3., 3., 6., 6., 8., 8., 8., 4., 4.],
                           [3., 3., 3., 6., 6., 8., 8., 8., 4., 4.],
                           [3., 3., 3., 6., 6., 8., 8., 8., 4., 4.],
                           [4., 4., 4., 8., 8., 4., 4., 4., 16., 16.],
                           [4., 4., 4., 8., 8., 4., 4., 4., 16., 16.]])

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
    W, Q = compute_objective_matrices(basis, basis_dimension, quad_model, derivative=False)
    expected_W = np.array([1., 0., 0., 2., 0.])
    expected_Q = np.array([[1., 0., 0., 2., 0.],
                           [0., 1/3, 0., 0., 2/3],
                           [0., 0., 1/5, 0., 0.],
                           [2., 0., 0., 4., 0.],
                           [0., 2/3, 0., 0., 4/3]])

    np.testing.assert_almost_equal(W, expected_W)
    np.testing.assert_almost_equal(Q, expected_Q)

    w = np.array([1., 2., 0., 3.])
    q = np.array([[1., 2., 3., 4.],
                  [2., 4., 6., 8.],
                  [3., 6., 8., 4.],
                  [4., 8., 4., 16.]])
    quad_model = [c, w, q]
    W, Q = compute_objective_matrices(basis, basis_dimension, quad_model, derivative=True)
    expected_W = np.array([1., 0., 0., 2., 6.])
    expected_Q = np.array([[1., 6., 0., 2., 8.],
                           [6., 32.33333333, 6., 12., 16.66666666],
                           [0., 6., 96.2, 0., 12.],
                           [2., 12., 0., 4., 16.],
                           [8., 16.66666666, 12., 16., 65.33333333]])

    np.testing.assert_almost_equal(W, expected_W)
    np.testing.assert_almost_equal(Q, expected_Q)