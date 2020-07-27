# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Test the basis module
"""
import unittest

import numpy as np

from pyrotor.data_analysis import nb_samples_is_sufficient
from pyrotor.data_analysis import compute_covariance


def test_nb_samples_is_sufficient():
    X = np.random.rand(10, 3)

    expected_boolean = True

    assert nb_samples_is_sufficient(X) == expected_boolean


def test_compute_covariance():
    X1 = [[1,0,1],
          [-1,3,2],
          [0,4,4],
          [3,3,6],
          [1,2,3],
          [100,-10, 90],
          [.1,1,1.1],
          [2,3,5],
          [4,8,12]]
    X2 = [[1,0,1],
          [-1,3,2],
          [0,4,4],
          [3,3,6],
          [1,2,3]]
    covariance1, precision1 = compute_covariance(X1)
    covariance2, precision2 = compute_covariance(X2)
    
    expected_covariance1 = np.array([[965.01333333, -125.01851852, 839.99481481],
                                     [-125.01851852, 21.13580247, -103.88271605],
                                     [839.99481481, -103.88271605, 736.11209877]])
    expected_precision1 = np.array([[0.01280836, -0.03140317, -0.01859481],
                                    [-0.03140317, 0.0788133 , 0.04741013],
                                    [-0.01859481, 0.04741013, 0.02881532]])
    expected_covariance2 = np.array([[965.01333333, -125.00851852, 839.98481481],
                                     [-125.00851852, 21.13580247, -103.89271605],
                                     [839.98481481, -103.89271605, 736.11209877]])
    expected_precision2 = np.array([[16.67949, 16.63522587, -16.68528413],
                                    [16.63522587, 16.74557453, -16.6191996 ],
                                    [-16.68528413, -16.6191996 , 16.69551627]])

    np.testing.assert_almost_equal(covariance1, expected_covariance1)
    np.testing.assert_almost_equal(covariance2, expected_covariance2)
    np.testing.assert_almost_equal(precision1, expected_precision1)
    np.testing.assert_almost_equal(precision2, expected_precision2)