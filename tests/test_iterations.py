# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Test the iterations module
"""

import unittest
import pytest
import mock
import numpy as np

from pyrotor.iterations import get_kappa_boundaries
from pyrotor.iterations import compute_kappa_min
from pyrotor.iterations import compute_kappa_max
from pyrotor.iterations import compute_kappa_mean
from pyrotor.iterations import compute_f
from pyrotor.iterations import compute_g
from pyrotor.iterations import binary_search_best_trajectory
from pyrotor.iterations import iterate_through_kappas


def test_get_kappa_boundaries():
    x = np.array([[1, 2], [1, 2]])
    Q = np.array([[1, 2], [3, 4]])
    W = np.array([2, 3])
    sigma_inverse = np.array([[1, 2], [3, 4]])
    c_weight = np.array([1, 1])
    expected_kappa_min = -2e-4
    expected_kappa_max = -2e2
    kappa_min, kappa_max = get_kappa_boundaries(x, Q, W,
                                                sigma_inverse, c_weight)
    assert kappa_min == expected_kappa_min
    assert kappa_max == expected_kappa_max


def test_compute_kappa_min():
    kappa_mean = 1
    expected_kappa_min = 1e-3
    kappa_min = compute_kappa_min(kappa_mean)
    assert kappa_min == expected_kappa_min


def test_compute_kappa_max():
    kappa_mean = 1
    expected_kappa_min = 1e3
    kappa_min = compute_kappa_max(kappa_mean)
    assert kappa_min == expected_kappa_min


def test_compute_kappa_mean():
    f_0 = 1
    g_0 = 2
    expected_kappa_mean = 0.5
    kappa_mean = compute_kappa_mean(f_0, g_0)
    assert kappa_mean == expected_kappa_mean


def test_compute_f():
    x = np.array([1, 2])
    sigma_inverse = np.array([[1, 2], [3, 4]])
    c_weight = np.array([1, 1])

    expected_f_0 = -7
    f_0 = compute_f(x, sigma_inverse, c_weight)
    assert f_0 == expected_f_0


def test_compute_g():
    x = np.array([1, 2])
    Q = np.array([[1, 2], [3, 4]])
    W = np.array([2, 3])
    expected_g_0 = 35
    g_0 = compute_g(x, Q, W)
    assert g_0 == expected_g_0


class TestTrajectoryIterations(unittest.TestCase):

    def setUp(self):
        self.trajectory = mock.Mock()
        self.i_call = 0

    def fake_compute_trajectory(self):
        self.i_call += 1
        if self.i_call == 2:
            self.trajectory.is_valid = True

    def test_binary_search_best_trajectory(self):

        self.trajectory.i_binary_search = 0
        # case 1: i < 0 -> ValueError as we can't find a solution to this
        # optimization
        with pytest.raises(ValueError):
            binary_search_best_trajectory(self.trajectory, -1, 5)

        # case 2:
        self.trajectory.is_valid = True
        self.trajectory.original_weights = [0, 1, 2]
        self.trajectory.kappas = [1, 2, 3]
        self.i_call = 0

        self.trajectory.compute_trajectory = self.fake_compute_trajectory
        binary_search_best_trajectory(self.trajectory, 2, 0)
        assert self.i_call == 1

        # FIXME: to test when required dependencies tested
        # # case 3:
        # i_call = 0
        # trajectory.is_valid = False
        # trajectory.compute_trajectory = self.fake_compute_trajectory
        # binary_search_best_trajectory(optimization, 2, 0)
        # assert i_call > 1

    def test_iterate_through_kappas(self):
        pass
