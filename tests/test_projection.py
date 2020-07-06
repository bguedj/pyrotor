# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Test the projection module
"""
import unittest
import numpy as np
import pandas as pd
from numpy.polynomial import legendre

from pyrotor.projection import trajectory_to_coef
from pyrotor.projection import trajectories_to_coefs
from pyrotor.projection import coef_to_trajectory
from pyrotor.projection import integrate_basis_legendre


def test_trajectory_to_coef():
    y = pd.DataFrame({"A": [1, 2, 3, 4, 5],
                      "B": [-4, -1, 4, 11, 20]})
    basis_dimensions = {"A": 3, "B": 2}
    basis = "legendre"
    expected_coef_A = pd.Series([3.5, 2.5, 0], name="A", dtype='float64')
    expected_coef_B = pd.Series([9, 15], name="B", dtype='float64')

    result = trajectory_to_coef(y, basis, basis_dimensions)
    result_A = result[0]
    result_B = result[1]

    pd.testing.assert_series_equal(expected_coef_A, result_A)
    pd.testing.assert_series_equal(expected_coef_B, result_B)


def test_trajectories_to_coefs():
    y = [pd.DataFrame({"A": [1, 2, 3, 4, 5]}),
         pd.DataFrame({"A": [-4, -1, 4, 11, 20]})]
    basis_dimensions = {"A": 2}
    basis = "legendre"
    expected_coefs_traj_1 = np.array([3.5, 2.5])
    expected_coefs_traj_2 = np.array([9, 15])

    result = trajectories_to_coefs(y, basis, basis_dimensions)
    result_1 = result[0]
    result_2 = result[1]

    np.testing.assert_almost_equal(result_1, expected_coefs_traj_1)
    np.testing.assert_almost_equal(result_2, expected_coefs_traj_2)


def test_coef_to_trajectory():
    coefs = [pd.Series([3.5, 2.5, 0], name="A", dtype='float64'),
             pd.Series([9, 15], name="B", dtype='float64')]
    duration = 5
    basis_dimensions = {"A": 3, "B": 2}
    basis = "legendre"

    expected_trajectory = pd.DataFrame({"A": [1, 2.25, 3.5, 4.75, 6],
                                        "B": [-6, 1.5, 9.0, 16.5, 24]},
                                       dtype='float64')

    result = coef_to_trajectory(coefs, duration, basis, basis_dimensions)

    pd.testing.assert_frame_equal(expected_trajectory, result)


def test_integrate_bases_legendre():
    expected_mean = np.array([1., 0., 0., 1., 0.])
    expected_dot_product = np.array([[1., 0., 0., 1., 0.],
                                     [0., 0.33333333, 0., 0., 0.33333333],
                                     [0., 0., 0.2, 0., 0.],
                                     [1., 0., 0., 1., 0.],
                                     [0., 0.33333333, 0., 0., 0.33333333]])
    basis_dimensions = {"A": 3, "B": 2}
    mean, dot_product = integrate_basis_legendre(basis_dimensions)

    np.testing.assert_almost_equal(mean, expected_mean)
    np.testing.assert_almost_equal(dot_product, expected_dot_product)
