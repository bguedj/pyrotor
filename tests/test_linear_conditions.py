import unittest
import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint

from pyrotor.linear_conditions import get_endpoints_matrix
from pyrotor.linear_conditions import format_endpoints
from pyrotor.linear_conditions import get_linear_endpoints


def test_get_endpoints_matrix():
    basis_dimension = {"A": 2}
    expected = np.array([[1., -1.],
                         [1.,  1.]])
    assert (expected == get_endpoints_matrix(basis_dimension)).all()

    basis_dimension = {"A": 3, "B": 2}
    expected = np.array([[1., -1.,  1.,  0.,  0.],
                         [0.,  0.,  0.,  1., -1.],
                         [1.,  1.,  1.,  0.,  0.],
                         [0.,  0.,  0.,  1.,  1.]])
    assert (expected == get_endpoints_matrix(basis_dimension)).all()


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
    basis_dimensions = {"A": 3, "B": 2}
    expected_phi = np.array([[1., -1.,  1.,  0.,  0.],
                    [0.,  0.,  0.,  1., -1.],
                    [1.,  1.,  1.,  0.,  0.],
                    [0.,  0.,  0.,  1.,  1.]])
    endpoints = {"A": {"start": 1, "end": 2, "delta": 1},
                 "B": {"start": 5, "end": 7, "delta": 2}}
    endpoints_delta = {'A': 1, 'B': 2}
    lb = np.array([0, 3, 1, 5])
    ub = np.array([2, 7, 3, 9])

    linear_endpoints, phi = get_linear_endpoints(basis_dimensions, endpoints)

    assert (phi == linear_endpoints.A).all()
    assert (lb == linear_endpoints.lb).all()
    assert (ub == linear_endpoints.ub).all()
    assert (phi == expected_phi).all()
