# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Test the basis module
"""
import unittest

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
                                    [-0.03140317, 0.0788133 , 0.04741013],
                                    [-0.01859481, 0.04741013, 0.02881532]])
    expected_covariance2 = np.array([[1.76, -0.31, 1.43],
                                     [-0.31, 1.84, 1.51],
                                     [1.43, 1.51, 2.96]])
    expected_precision2 = np.array([[16.94567835, 16.46721969, -16.58710195],
                                    [16.46721969, 16.93711533, -16.59566497],
                                    [-16.58710195, -16.59566497, 16.81723307]])

    np.testing.assert_almost_equal(covariance1, expected_covariance1)
    np.testing.assert_almost_equal(covariance2, expected_covariance2)
    np.testing.assert_almost_equal(precision1, expected_precision1)
    np.testing.assert_almost_equal(precision2, expected_precision2)


def test_compute_cost():
      # TODO: Test in case quad_model is a path to a pickle model
      trajectory1 = pd.DataFrame({"A": [1, 1, 1],
                                  "B": [1, 1, 1]})
      trajectory2 = pd.DataFrame({"A": [3, 5, 7],
                                  "B": [3, 2, 1]})
      trajectory3 = pd.DataFrame({"A": [0, 0, 1],
                                  "B": [0, 1, 2]})
      trajectories = [trajectory1, trajectory2, trajectory3]
      w = - 2 * np.ones(2)
      q = np.eye(2)
      quad_model = [w, q]

      trajectories_cost = compute_cost(trajectories, quad_model)

      expected_trajectories_cost = np.array([-6, 55, -2])

      np.testing.assert_equal(trajectories_cost, expected_trajectories_cost)


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

      best_trajectories, best_cost = select_trajectories(trajectories, trajectories_cost, trajectories_nb)

      expected_best_trajectories = [trajectory1, trajectory3]
      expected_best_cost = np.array([-6, -2])

      assert best_trajectories == expected_best_trajectories
      np.testing.assert_equal(best_cost, expected_best_cost)


def test_compute_weights():
      trajectories_cost = np.array([-6, 55, -2])

      def g(x): return x + 10

      weights1 = compute_weights(trajectories_cost, f=g)
      weights2 = compute_weights(trajectories_cost)

      expected_weights1 = np.array([4/77, 65/77, 8/77])
      expected_weights2 = np.exp(-trajectories_cost)
      expected_weights2 /= np.sum(np.exp(-trajectories_cost))

      np.testing.assert_almost_equal(weights1, expected_weights1)
      np.testing.assert_almost_equal(weights2, expected_weights2)
