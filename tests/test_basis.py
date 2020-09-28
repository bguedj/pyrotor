# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Test the basis module
"""
import unittest

import numpy as np
from numpy.polynomial import legendre

from pyrotor.basis import compute_legendre_features
from pyrotor.basis import compute_legendre_mean_dot_product
from pyrotor.basis import compute_legendre_mean_derivatives
from pyrotor.basis import compute_legendre_dot_product_derivatives


def test_compute_legendre_features():
    expected_mean_1 = np.array([1., 0., 0., 1., 0.])
    expected_mean_2 = np.array([0, 2, 0, 0, 2])
    expected_mean = np.concatenate((expected_mean_1, expected_mean_2))
    expected_dot_product_11 = np.array([[1., 0., 0., 1., 0.],
                                        [0., 0.33333333, 0., 0., 0.33333333],
                                        [0., 0., 0.2, 0., 0.],
                                        [1., 0., 0., 1., 0.],
                                        [0., 0.33333333, 0., 0., 0.33333333]])
    expected_dot_product_12 = np.array([[0., 2., 0., 0., 2.],
                                        [0., 0., 2., 0., 0.],
                                        [0., 0., 0., 0., 0.],
                                        [0., 2., 0., 0., 2.],
                                        [0., 0., 2., 0., 0.]])
    expected_dot_product_22 = np.array([[0., 0., 0., 0., 0.],
                                        [0., 4., 0., 0., 4.],
                                        [0., 0., 12., 0., 0.],
                                        [0., 0., 0., 0., 0.],
                                        [0., 4., 0., 0., 4.]])
    expected_dot_product_1 = np.concatenate((expected_dot_product_11, expected_dot_product_12), axis=1)
    expected_dot_product_2 = np.concatenate((expected_dot_product_12.T, expected_dot_product_22), axis=1)
    expected_dot_product = np.concatenate((expected_dot_product_1, expected_dot_product_2), axis=0)
    basis_dimension = {"A": 3, "B": 2}
    mean, dot_product = compute_legendre_features(basis_dimension, derivative=True)

    np.testing.assert_almost_equal(mean, expected_mean)
    np.testing.assert_almost_equal(dot_product, expected_dot_product)


def test_compute_legendre_mean_dot_product():
    expected_mean = np.array([1., 0., 0., 1., 0.])
    expected_dot_product = np.array([[1., 0., 0., 1., 0.],
                                     [0., 0.33333333, 0., 0., 0.33333333],
                                     [0., 0., 0.2, 0., 0.],
                                     [1., 0., 0., 1., 0.],
                                     [0., 0.33333333, 0., 0., 0.33333333]])
    basis_dimension = {"A": 3, "B": 2}
    mean, dot_product = compute_legendre_mean_dot_product(basis_dimension)

    np.testing.assert_almost_equal(mean, expected_mean)
    np.testing.assert_almost_equal(dot_product, expected_dot_product)


def test_compute_legendre_mean_derivatives():
    expected_mean = np.array([0, 2, 0, 0, 2])
    basis_dimension = {"A": 3, "B": 2}
    mean = compute_legendre_mean_derivatives(basis_dimension)

    np.testing.assert_almost_equal(mean, expected_mean)


def test_compute_legendre_dot_product_derivatives():
    expected_dot_product_12 = np.array([[0., 2., 0., 0., 2.],
                                        [0., 0., 2., 0., 0.],
                                        [0., 0., 0., 0., 0.],
                                        [0., 2., 0., 0., 2.],
                                        [0., 0., 2., 0., 0.]])
    expected_dot_product_22 = np.array([[0., 0., 0., 0., 0.],
                                        [0., 4., 0., 0., 4.],
                                        [0., 0., 12., 0., 0.],
                                        [0., 0., 0., 0., 0.],
                                        [0., 4., 0., 0., 4.]])
    basis_dimension = {"A": 3, "B": 2}
    dot_product_12, dot_product_22 = compute_legendre_dot_product_derivatives(basis_dimension)

    np.testing.assert_almost_equal(dot_product_12, expected_dot_product_12)
    np.testing.assert_almost_equal(dot_product_22, expected_dot_product_22)
