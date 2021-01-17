# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Test the basis module
"""
import unittest

import numpy as np
from numpy.polynomial import legendre

from pyrotor.basis import compute_basis_features
from pyrotor.basis import compute_legendre_integral_dot_product
from pyrotor.basis import compute_legendre_integral_derivatives
from pyrotor.basis import compute_legendre_dot_product_derivatives
from pyrotor.basis import compute_bspline_integral_dot_product
from pyrotor.basis import compute_bspline_integral_derivatives
from pyrotor.basis import compute_bspline_dot_product_derivatives


def test_compute_basis_features():
    # Test Legendre
    basis = 'legendre'
    basis_features = {}
    expected_integral_1 = np.array([1., 0., 0., 1., 0.])
    expected_integral_2 = np.array([0, 2, 0, 0, 2])
    expected_integral = np.concatenate((expected_integral_1, expected_integral_2))
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
    integral, dot_product = compute_basis_features(basis, basis_features, basis_dimension, derivative=True)

    np.testing.assert_almost_equal(integral, expected_integral)
    np.testing.assert_almost_equal(dot_product, expected_dot_product)

    # Test B-spline
    basis = 'bspline'
    basis_features = {'knots': [.33, .66], 'A': 2, 'B': 1}
    basis_dimension = {"A": 5, "B": 4}
    expected_integral_1 = np.array([1/9, 2/9, 1/3, 2/9, 1/9, 1/6, 1/3, 1/3, 1/6])
    expected_integral_2 = np.array([-1., 0., 0., 0., 1., -1., 0., 0., 1.])
    expected_integral = np.concatenate((expected_integral_1, expected_integral_2))
    expected_dot_product_11 = np.array([[.066, .0385, .0055, 0., 0., .0825, .0275, 0., 0.],
                                        [.0385, .11, .06879104, .00270896, 0., .06875 , .1375, .01375 , 0.],
                                        [.0055, .06879104, .1835821 , .06970895, .00575124, .01375 , .15145522, .15375 , .01437811],
                                        [0., .00270896, .06970895, .11133333, .03958209, 0., .01354478, .13916667, .07062189],
                                        [0., 0., .00575124, .03958209, .068 , 0., 0., .02833333, .085],
                                        [.0825, .06875 , .01375 , 0., 0., .11, .055 , 0., 0.],
                                        [.0275, .1375, .15145522, .01354478, 0., .055 , .22, .055 , 0.],
                                        [0., .01375 , .15375 , .13916667, .02833333, 0., .055 , .22333333, .05666667],
                                        [0., 0., .01437811, .07062189, .085 , 0., 0., .05666667, .11333333]])
    expected_dot_product_12 = np.array([[-.5,  4.16666667e-01,  8.33333333e-02, 0.,  0., -3.33333333e-01, 3.33333333e-01,  0.,  0.],
                                        [-4.16666667e-01, 0.,  3.75621891e-01, 4.10447761e-02,  0., -.5, 3.33333333e-01,  1.66666667e-01,  0.],
                                        [-8.33333333e-02, -3.75621891e-01, 0., 3.74378109e-01,  8.45771144e-02, -1.66666667e-01, -5.02487562e-01,  .5,  1.69154229e-01],
                                        [ 0., -4.10447761e-02, -3.74378109e-01, -1.23871485e-11,  4.15422886e-01,  0., -1.64179104e-01, -3.33333332e-01,  4.97512438e-01],
                                        [ 0.,  0., -8.45771144e-02, -4.15422886e-01,  .5,  0., 0., -3.33333333e-01,  3.33333333e-01],
                                        [-6.66666667e-01,  .5,  1.66666667e-01, 0.,  0., -.5, .5,  0.,  0.],
                                        [-3.33333333e-01, -3.33333333e-01,  5.02487562e-01, 1.64179104e-01,  0., -.5, 0.,  .5,  0.],
                                        [ 0., -1.66666667e-01, -.5, 3.33333333e-01,  3.33333333e-01,  0., -.5,  0.,  .5],
                                        [ 0.,  0., -1.69154229e-01, -4.97512438e-01,  6.66666667e-01,  0., 0., -.5,  .5]])
    expected_dot_product_22 = np.array([[ 4.04040404, -3.03030303, -1.01010101,  0.,  0., 3.03030303, -3.03030303,  0.,  0.],
                                        [-3.03030303,  4.04040404, -0.51258857, -0.49751244,  0., -1.51515152,  3.03030303, -1.51515152,  0.],
                                        [-1.01010101, -0.51258857,  3.0152269 , -0.49751244, -0.99502488, -1.51515152,  1.49253731,  1.51515152, -1.49253731],
                                        [ 0., -0.49751244, -0.49751244,  3.92156863, -2.92654375, 0., -1.49253731,  2.94117647, -1.44863916],
                                        [ 0.,  0., -0.99502488, -2.92654375,  3.92156863, 0.,  0., -2.94117647,  2.94117647],
                                        [ 3.03030303, -1.51515152, -1.51515152,  0.,  0., 3.03030303, -3.03030303,  0.,  0.],
                                        [-3.03030303,  3.03030303,  1.49253731, -1.49253731,  0., -3.03030303,  6.06060606, -3.03030303,  0.],
                                        [ 0., -1.51515152,  1.51515152,  2.94117647, -2.94117647, 0., -3.03030303,  5.97147949, -2.94117647],
                                        [ 0.,  0., -1.49253731, -1.44863916,  2.94117647, 0.,  0., -2.94117647,  2.94117647]])
    expected_dot_product_1 = np.concatenate((expected_dot_product_11, expected_dot_product_12), axis=1)
    expected_dot_product_2 = np.concatenate((expected_dot_product_12.T, expected_dot_product_22), axis=1)
    expected_dot_product = np.concatenate((expected_dot_product_1, expected_dot_product_2), axis=0)
    integral, dot_product = compute_basis_features(basis, basis_features, basis_dimension, derivative=True)

    # Use decimal=2 due to numerical integration which is not exact 
    np.testing.assert_almost_equal(integral, expected_integral, decimal=2)
    np.testing.assert_almost_equal(dot_product, expected_dot_product)


def test_compute_legendre_integral_dot_product():
    expected_integral = np.array([1., 0., 0., 1., 0.])
    expected_dot_product = np.array([[1., 0., 0., 1., 0.],
                                     [0., 0.33333333, 0., 0., 0.33333333],
                                     [0., 0., 0.2, 0., 0.],
                                     [1., 0., 0., 1., 0.],
                                     [0., 0.33333333, 0., 0., 0.33333333]])
    basis_dimension = {"A": 3, "B": 2}
    integral, dot_product = compute_legendre_integral_dot_product(basis_dimension)

    np.testing.assert_almost_equal(integral, expected_integral)
    np.testing.assert_almost_equal(dot_product, expected_dot_product)


def test_compute_legendre_integral_derivatives():
    expected_integral = np.array([0, 2, 0, 0, 2])
    basis_dimension = {"A": 3, "B": 2}
    integral = compute_legendre_integral_derivatives(basis_dimension)

    np.testing.assert_almost_equal(integral, expected_integral)


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

def test_compute_bspline_integral_dot_product():
    expected_integral = np.array([1/9, 2/9, 1/3, 2/9, 1/9, 1/6, 1/3, 1/3, 1/6])
    expected_dot_product = np.array([[.066, .0385, .0055, 0., 0., .0825, .0275, 0., 0.],
                                     [.0385, .11, .06879104, .00270896, 0., .06875 , .1375, .01375 , 0.],
                                     [.0055, .06879104, .1835821 , .06970895, .00575124, .01375 , .15145522, .15375 , .01437811],
                                     [0., .00270896, .06970895, .11133333, .03958209, 0., .01354478, .13916667, .07062189],
                                     [0., 0., .00575124, .03958209, .068 , 0., 0., .02833333, .085],
                                     [.0825, .06875 , .01375 , 0., 0., .11, .055 , 0., 0.],
                                     [.0275, .1375, .15145522, .01354478, 0., .055 , .22, .055 , 0.],
                                     [0., .01375 , .15375 , .13916667, .02833333, 0., .055 , .22333333, .05666667],
                                     [0., 0., .01437811, .07062189, .085 , 0., 0., .05666667, .11333333]])

    basis_features = {'knots': [.33, .66], 'A': 2, 'B': 1}
    basis_dimension = {"A": 5, "B": 4}
    integral, dot_product = compute_bspline_integral_dot_product(basis_features, basis_dimension)

    # Use decimal=2 due to numerical integration which is not exact
    np.testing.assert_almost_equal(integral, expected_integral, decimal=2)
    np.testing.assert_almost_equal(dot_product, expected_dot_product)

def test_compute_bspline_integral_derivatives():
    expected_integral_deriv = np.array([-1., 0., 1., -1., 0., 0., 1., -1., 0., 1.])

    basis_dimension = {"A": 3, "B": 4, "C": 3}
    integral_deriv = compute_bspline_integral_derivatives(basis_dimension)

    np.testing.assert_almost_equal(integral_deriv, expected_integral_deriv)

def test_compute_bspline_dot_product_derivatives():
    expected_dot_product_12 = np.array([[-.5,  4.16666667e-01,  8.33333333e-02, 0.,  0., -3.33333333e-01, 3.33333333e-01,  0.,  0.],
                                        [-4.16666667e-01, 0.,  3.75621891e-01, 4.10447761e-02,  0., -.5, 3.33333333e-01,  1.66666667e-01,  0.],
                                        [-8.33333333e-02, -3.75621891e-01, 0., 3.74378109e-01,  8.45771144e-02, -1.66666667e-01, -5.02487562e-01,  .5,  1.69154229e-01],
                                        [ 0., -4.10447761e-02, -3.74378109e-01, -1.23871485e-11,  4.15422886e-01,  0., -1.64179104e-01, -3.33333332e-01,  4.97512438e-01],
                                        [ 0.,  0., -8.45771144e-02, -4.15422886e-01,  .5,  0., 0., -3.33333333e-01,  3.33333333e-01],
                                        [-6.66666667e-01,  .5,  1.66666667e-01, 0.,  0., -.5, .5,  0.,  0.],
                                        [-3.33333333e-01, -3.33333333e-01,  5.02487562e-01, 1.64179104e-01,  0., -.5, 0.,  .5,  0.],
                                        [ 0., -1.66666667e-01, -.5, 3.33333333e-01,  3.33333333e-01,  0., -.5,  0.,  .5],
                                        [ 0.,  0., -1.69154229e-01, -4.97512438e-01,  6.66666667e-01,  0., 0., -.5,  .5]])
    expected_dot_product_22 = np.array([[ 4.04040404, -3.03030303, -1.01010101,  0.,  0., 3.03030303, -3.03030303,  0.,  0.],
                                        [-3.03030303,  4.04040404, -0.51258857, -0.49751244,  0., -1.51515152,  3.03030303, -1.51515152,  0.],
                                        [-1.01010101, -0.51258857,  3.0152269 , -0.49751244, -0.99502488, -1.51515152,  1.49253731,  1.51515152, -1.49253731],
                                        [ 0., -0.49751244, -0.49751244,  3.92156863, -2.92654375, 0., -1.49253731,  2.94117647, -1.44863916],
                                        [ 0.,  0., -0.99502488, -2.92654375,  3.92156863, 0.,  0., -2.94117647,  2.94117647],
                                        [ 3.03030303, -1.51515152, -1.51515152,  0.,  0., 3.03030303, -3.03030303,  0.,  0.],
                                        [-3.03030303,  3.03030303,  1.49253731, -1.49253731,  0., -3.03030303,  6.06060606, -3.03030303,  0.],
                                        [ 0., -1.51515152,  1.51515152,  2.94117647, -2.94117647, 0., -3.03030303,  5.97147949, -2.94117647],
                                        [ 0.,  0., -1.49253731, -1.44863916,  2.94117647, 0.,  0., -2.94117647,  2.94117647]])

    basis_features = {'knots': [.33, .66], 'A': 2, 'B': 1}
    basis_dimension = {"A": 5, "B": 4}
    dot_product_12, dot_product_22 = compute_bspline_dot_product_derivatives(basis_features, basis_dimension)

    np.testing.assert_almost_equal(dot_product_12, expected_dot_product_12)
    np.testing.assert_almost_equal(dot_product_22, expected_dot_product_22)
