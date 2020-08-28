# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Test the basis module
"""
import unittest
from pytest import raises

import numpy as np
import pandas as pd

from pyrotor.data_analysis import nb_samples_is_sufficient
from pyrotor.data_analysis import compute_covariance
from pyrotor.data_analysis import select_trajectories
from pyrotor.data_analysis import compute_weights


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
                                    [-0.03140317, 0.0788133, 0.04741013],
                                    [-0.01859481, 0.04741013, 0.02881532]])
    expected_covariance2 = np.array([[1.76, -0.3099437, 1.43000351],
                                     [-0.3099437, 1.84, 1.51],
                                     [1.43000351, 1.51, 2.96]])
    expected_precision2 = np.array([[ 16.91494836, 16.43864249, -16.55694603],
                                    [ 16.43864249,  16.90863849, -16.56548905],
                                    [-16.55694603, -16.56548905,  16.7872707 ]])

    np.testing.assert_almost_equal(covariance1, expected_covariance1)
    np.testing.assert_almost_equal(covariance2, expected_covariance2)
    np.testing.assert_almost_equal(precision1, expected_precision1)
    np.testing.assert_almost_equal(precision2, expected_precision2)

    X3 = [[k for k in range(10)], [k+1 for k in range(10)]]
    with raises(ValueError):
        compute_covariance(X3)
    

def test_select_trajectories():
    trajectory1 = pd.DataFrame({"A": [1, 1, 1],
                               "B": [1, 1, 1]})
    trajectory2 = pd.DataFrame({"A": [3, 5, 7],
                               "B": [3, 2, 1]})
    trajectory3 = pd.DataFrame({"A": [0, 0, 1],
                               "B": [0, 1, 2]})
    trajectories = [trajectory1, trajectory2, trajectory3]
    trajectories_cost = np.array([-6, 55, -2])
    trajectories_nb = 2

    best_trajectories = select_trajectories(trajectories, trajectories_cost,
                                            trajectories_nb)

    expected_best_trajectories = [trajectory1, trajectory3]

    pd.testing.assert_frame_equal(best_trajectories[0],
                                  expected_best_trajectories[0])
    pd.testing.assert_frame_equal(best_trajectories[1],
                                  expected_best_trajectories[1])
    assert len(best_trajectories) == len(expected_best_trajectories)


def test_compute_weights():
    trajectories_cost = np.array([-6, 55, -2])

    def g(x): return x + 10

    weights1 = compute_weights(trajectories_cost, weight_fonction=g)
    weights2 = compute_weights(trajectories_cost)

    expected_weights1 = np.array([4/77, 65/77, 8/77])
    expected_weights2 = np.exp(trajectories_cost-np.max(trajectories_cost))
    expected_weights2 /= np.sum(expected_weights2)

    np.testing.assert_almost_equal(weights1, expected_weights1)
    np.testing.assert_almost_equal(weights2, expected_weights2)
