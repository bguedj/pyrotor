# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Test the cost functions module
"""

import pytest
import numpy as np
import pandas as pd

from pyrotor.cost_functions import compute_f
from pyrotor.cost_functions import compute_g
from pyrotor.cost_functions import compute_cost
from pyrotor.cost_functions import compute_cost_by_time
from pyrotor.cost_functions import compute_trajectories_cost


def test_compute_f():
    x = np.array([1, 2])
    sigma_inverse = np.array([[1, 2], [3, 4]])
    c_weight = np.array([1, 1])

    expected_f_0 = -7
    f_0 = compute_f(x, sigma_inverse, c_weight)
    assert f_0 == expected_f_0


def test_compute_g():
    # TODO: Test when model is sklearn model
    x = np.array([1, 2])
    Q = np.array([[1, 2], [3, 4]])
    W = np.array([2, 3])
    model = [W, Q]
    basis = 'legendre'
    basis_dimension = {"A": 1, "B": 1}
    basis_features = basis_dimension
    independent_variable = {'start': 0, 'end': 1, 'points_nb': 2}
    extra_info = {'basis': basis, 
                  'basis_dimension': basis_dimension,
                  'basis_features': basis_features, 
                  'independent_variable': independent_variable}
    expected_g_0 = 35
    g_0 = compute_g(x, model, extra_info)
    assert g_0 == expected_g_0


def test_compute_cost():
    # TODO: Test when model is sklearn model
    trajectory = pd.DataFrame({"A": [1, 3], "B": [2, 4]})
    quadratic_part = np.array([[1, 0], [2, 3]])
    linear_part = np.array([2, 1])
    constant_part = 8
    model = (constant_part, linear_part, quadratic_part)
    independent_variable = {'start': 0, 'end': 1, 'points_nb': 2}
    cost = compute_cost(trajectory, model, independent_variable)
    assert cost == 64


def test_compute_cost_by_time():
    # TODO: Test when model is sklearn model
    trajectory = pd.DataFrame({"A": [1, 3], "B": [2, 4]})
    quadratic_part = np.array([[1, 0], [2, 3]])
    linear_part = np.array([2, 1])
    constant_part = 8
    model = (constant_part, linear_part, quadratic_part)

    cost_by_time = compute_cost_by_time(trajectory, model)
    expected_cost_by_time = np.array([29, 99])

    np.testing.assert_equal(cost_by_time, expected_cost_by_time)


def test_compute_trajectories_cost():
    # TODO: Test when model is sklearn model
    trajectory1 = pd.DataFrame({"A": [1, 1, 1],
                                "B": [1, 1, 1]})
    trajectory2 = pd.DataFrame({"A": [3, 5, 7],
                                "B": [3, 2, 1]})
    trajectory3 = pd.DataFrame({"A": [0, 0, 1],
                                "B": [0, 1, 2]})
    trajectories = [trajectory1, trajectory2, trajectory3]
    w = - 2 * np.ones(2)
    q = np.eye(2)
    c = 1
    model = [c, w, q]
    independent_variable = {'start': 0, 'end': 2, 'points_nb': 3}

    trajectories_cost = compute_trajectories_cost(trajectories, model, independent_variable)

    expected_trajectories_cost = np.array([-2, 37, .5])

    np.testing.assert_equal(trajectories_cost, expected_trajectories_cost)
