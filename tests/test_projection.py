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
from pyrotor.projection import compute_weighted_coef
from pyrotor.projection import coef_to_trajectory


def test_trajectory_to_coef():
    # Test Legendre
    y = pd.DataFrame({"A": [1, 2, 3, 4, 5],
                      "B": [-4, -1, 4, 11, 20]})
    basis_dimension = {"A": 3, "B": 2}
    basis_features = basis_dimension
    basis = "legendre"
    expected_coef = np.array([3., 2., 0., 6., 12.], dtype='float64')

    result = trajectory_to_coef(y, basis, basis_features, basis_dimension)

    np.testing.assert_almost_equal(expected_coef, result)

    # Test B-spline
    x = np.linspace(0, 1, 20)
    y = pd.DataFrame({"A": x,
                      "B": x**2})
    basis_features = {"knots": [.25, .5, .75], "A": 2, "B": 3}
    basis_dimension = {"A": 6, "B": 7}
    basis = "bspline"
    expected_coef = np.array([0., .125, .375, .625, .875, 1.,
                              0., 0., 4.16666667e-02, 2.29166667e-01, 5.41666667e-01, 8.33333333e-01, 1.], dtype='float64')

    result = trajectory_to_coef(y, basis, basis_features, basis_dimension)

    np.testing.assert_almost_equal(expected_coef, result)


def test_trajectories_to_coefs():
    # Test Legendre
    y = [pd.DataFrame({"A": [1, 2, 3, 4, 5]}),
         pd.DataFrame({"A": [-4, -1, 4, 11, 20]})]
    basis_dimension = {"A": 2}
    basis_features = basis_dimension
    basis = "legendre"
    expected_coefs_traj_1 = np.array([3., 2.])
    expected_coefs_traj_2 = np.array([6., 12.])

    n_jobs = None
    result = trajectories_to_coefs(y, basis, basis_features, basis_dimension, n_jobs)
    result_1 = result[0]
    result_2 = result[1]

    np.testing.assert_almost_equal(result_1, expected_coefs_traj_1)
    np.testing.assert_almost_equal(result_2, expected_coefs_traj_2)

    # Test B-spline
    x = np.linspace(0, 1, 20)
    y = [pd.DataFrame({"A": x}),
         pd.DataFrame({"A": 2 * x})]
    basis_features = {"knots": [.25, .5, .75], "A": 2}
    basis_dimension = {"A": 6}
    basis = "bspline"
    expected_coefs_traj_1 = np.array([0., .125, .375, .625, .875, 1.])
    expected_coefs_traj_2 = np.array([0., .25, .75, 1.25, 1.75, 2.])

    n_jobs = None
    result = trajectories_to_coefs(y, basis, basis_features, basis_dimension, n_jobs)
    result_1 = result[0]
    result_2 = result[1]

    np.testing.assert_almost_equal(result_1, expected_coefs_traj_1)
    np.testing.assert_almost_equal(result_2, expected_coefs_traj_2)


def test_compute_weighted_coef():
    coef1 = pd.Series([1, 2, 0], dtype='float64')
    coef2 = pd.Series([-2, 0, 1], dtype='float64')
    coefs = [coef1, coef2]
    weights = np.array([-1/3, 2/3])
    basis_dimension = {"A": 1, "B": 2}
    c_weight = compute_weighted_coef(coefs, weights, basis_dimension)

    expected_c_weight = np.array([-5/3, -2/3, 2/3])

    np.testing.assert_almost_equal(c_weight, expected_c_weight)


def test_coef_to_trajectory():
    # Test Legendre
    coefs = [pd.Series([3.5, 2.5, 0], name="A", dtype='float64'),
             pd.Series([9, 15], name="B", dtype='float64')]
    duration = 5
    basis_dimension = {"A": 3, "B": 2}
    basis_features = basis_dimension
    basis = "legendre"

    expected_trajectory = pd.DataFrame({"A": [1, 2.25, 3.5, 4.75, 6],
                                        "B": [-6, 1.5, 9.0, 16.5, 24]},
                                       dtype='float64')

    result = coef_to_trajectory(coefs, duration, basis, basis_features, basis_dimension)

    pd.testing.assert_frame_equal(expected_trajectory, result)
    
    # Test B-spline
    coefs = [pd.Series([0., .125, .375, .625, .875, 1.], name="A", dtype='float64'),
             pd.Series([0, 0, 4.16666667e-02, 2.29166667e-01, 5.41666667e-01, 8.33333333e-01, 1.], name="B", dtype='float64')]
    duration = 20
    basis_features = {"knots": [.25, .5, .75], "A": 2, "B": 3}
    basis_dimension = {"A": 6, "B": 7}
    basis = "bspline"

    x = np.linspace(0, 1, 20)
    expected_trajectory = pd.DataFrame({"A": x,
                                        "B": x**2},
                                        dtype='float64')

    result = coef_to_trajectory(coefs, duration, basis, basis_features, basis_dimension)

    pd.testing.assert_frame_equal(expected_trajectory, result)
