# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Test the iterations module
"""
import pytest
import mock
import numpy as np

from pyrotor.iterations import binary_search_best_trajectory
from pyrotor.iterations import iterate

optimization = None
i_call = 0


def fake_compute_trajectory():
    global i_call
    global optimization
    i_call += 1
    if i_call == 2:
        optimization.is_valid = True


def test_binary_search_best_trajectory():
    global i_call
    global optimization

    optimization = mock.Mock()
    optimization.i_binary_search = 0
    # case 1: i < 0 -> ValueError as we can't find a solution to this
    # optimization
    with pytest.raises(ValueError):
        binary_search_best_trajectory(optimization, -1, 5)

    # case 2:
    optimization.is_valid = True
    optimization.original_weights = [0, 1, 2]
    optimization.kappas = [1, 2, 3]
    i_call = 0

    optimization.compute_trajectory = fake_compute_trajectory
    binary_search_best_trajectory(optimization, 2, 0)
    assert i_call == 1

    # case 3:
    i_call = 0
    optimization.is_valid = False
    optimization.compute_trajectory = fake_compute_trajectory
    binary_search_best_trajectory(optimization, 2, 0)
    assert i_call > 1


def test_iterate():
    pass
