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
    # TODO: Test when quad_model is sklearn model
    # Test Legendre
    basis = 'legendre'
    basis_features = {}
    basis_dimension = {"A": 3, "B": 2}
    c = 0
    w = np.array([1., 2.])
    q = np.array([[1., 2.],
                  [2., 4.]])
    quad_model = [c, w, q]
    W, Q = compute_objective_matrices(basis, basis_features, basis_dimension, quad_model, derivative=False)
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
    W, Q = compute_objective_matrices(basis, basis_features, basis_dimension, quad_model, derivative=True)
    expected_W = np.array([1., 0., 0., 2., 6.])
    expected_Q = np.array([[1., 6., 0., 2., 8.],
                           [6., 32.33333333, 6., 12., 16.66666666],
                           [0., 6., 96.2, 0., 12.],
                           [2., 12., 0., 4., 16.],
                           [8., 16.66666666, 12., 16., 65.33333333]])

    np.testing.assert_almost_equal(W, expected_W)
    np.testing.assert_almost_equal(Q, expected_Q)

    # Test B-spline
    basis = 'bspline'
    basis_features = {'knots': [.33, .66], 'A': 2, 'B': 1}
    basis_dimension = {"A": 5, "B": 4}
    W, Q = compute_objective_matrices(basis, basis_features, basis_dimension, quad_model, derivative=False)
    expected_W = np.array([1/9, 2/9, 1/3, 2/9, 1/9, 1/3, 2/3, 2/3, 1/3])
    expected_Q = np.array([[.066, .0385, .0055, 0., 0.,.165, .055, 0., 0.],
                           [.0385, .11, .06879104, .00270896, 0., .1375, .275, .0275, 0.],
                           [.0055, .06879104, .1835821 , .06970895, .00575124, .0275, .30291045, .3075, .02875622],
                           [0., .00270896, .06970895, .11133333, .03958209, 0., .02708955, .27833333, .14124378],
                           [0., 0., .00575124, .03958209, .068, 0., 0., .05666667, .17],
                           [.165, .1375, .0275, 0., 0., .44, .22, 0., 0.],
                           [.055, .275, .30291045, .02708955, 0., .22, .88, .22, 0.],
                           [0., .0275, .3075, .27833333, .05666667, 0., .22, .89333333, .22666667],
                           [0., 0., .02875622, .14124378, .17, 0., 0., 0.22666667, 0.45333333]])

    # Use decimal=2 due to numerical integration which is not exact
    np.testing.assert_almost_equal(W, expected_W, decimal=2)
    np.testing.assert_almost_equal(Q, expected_Q)