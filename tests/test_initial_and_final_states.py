import unittest
import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint

from pyrotor.initial_and_final_states import build_matrix_endpoints
from pyrotor.initial_and_final_states import compute_weighted_average_endpoints
from pyrotor.initial_and_final_states import format_endpoints
from pyrotor.initial_and_final_states import get_linear_endpoints


def test_build_matrix_endpoints():
    basis_dimension = {"A": 2}
    expected = np.array([[1., -1.],
                         [1.,  1.]])
    assert (expected == build_matrix_endpoints(basis_dimension)).all()

    basis_dimension = {"A": 3, "B": 2}
    expected = np.array([[1., -1.,  1.,  0.,  0.],
                         [0.,  0.,  0.,  1., -1.],
                         [1.,  1.,  1.,  0.,  0.],
                         [0.,  0.,  0.,  1.,  1.]])
    assert (expected == build_matrix_endpoints(basis_dimension)).all()


def test_compute_weighted_average_endpoints():
    endpoints = {'A': {}, 'B': {}}
    omega = np.array([1, 2])
    ref_trajectories = [pd.DataFrame({'A': [1, 4], 'B': [1, 1]}),
                        pd.DataFrame({'A': [1, 1], 'B': [7, 10]})]
    expected = {'A': {'start': 1, 'end': 2},
                'B': {'start': 5, 'end': 7}}
    assert expected == compute_weighted_average_endpoints(endpoints,
                                                          omega,
                                                          ref_trajectories)


def test_format_endpoints():
    phi = np.array([[1., -1.,  1.,  0.,  0.],
                    [0.,  0.,  0.,  1., -1.],
                    [1.,  1.,  1.,  0.,  0.],
                    [0.,  0.,  0.,  1.,  1.]])
    endpoints = {'A': {'start': 1, 'end': 2, 'delta': 1},
                 'B': {'start': 5, 'end': 7, 'delta': 2}}
    lb = np.array([0, 3, 1, 5])
    ub = np.array([2, 7, 3, 9])
    result = format_endpoints(phi, endpoints)

    assert (phi == result.A).all()
    assert (lb == result.lb).all()
    assert (ub == result.ub).all()


def test_get_linear_endpoints():
    basis_dimension = {"A": 3, "B": 2}
    omega = np.array([1, 2])
    ref_trajectories = [pd.DataFrame({'A': [1, 4], 'B': [1, 1]}),
                        pd.DataFrame({'A': [1, 1], 'B': [7, 10]})]
    phi = np.array([[1., -1.,  1.,  0.,  0.],
                    [0.,  0.,  0.,  1., -1.],
                    [1.,  1.,  1.,  0.,  0.],
                    [0.,  0.,  0.,  1.,  1.]])
    endpoints_delta = {'A': 1, 'B': 2}
    lb = np.array([0, 3, 1, 5])
    ub = np.array([2, 7, 3, 9])

    result = get_linear_endpoints(basis_dimension, omega,
                                  ref_trajectories, endpoints_delta)

    assert (phi == result.A).all()
    assert (lb == result.lb).all()
    assert (ub == result.ub).all()
