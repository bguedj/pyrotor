# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Test the basis module
"""
import unittest

import numpy as np
from numpy.polynomial import legendre

from pyrotor.basis import basis_legendre


def test_basis_legendre():
    expected_mean = np.array([1., 0., 0., 1., 0.])
    expected_dot_product = np.array([[1., 0., 0., 1., 0.],
                                     [0., 0.33333333, 0., 0., 0.33333333],
                                     [0., 0., 0.2, 0., 0.],
                                     [1., 0., 0., 1., 0.],
                                     [0., 0.33333333, 0., 0., 0.33333333]])
    basis_dimension = {"A": 3, "B": 2}
    mean, dot_product = basis_legendre(basis_dimension)

    np.testing.assert_almost_equal(mean, expected_mean)
    np.testing.assert_almost_equal(dot_product, expected_dot_product)