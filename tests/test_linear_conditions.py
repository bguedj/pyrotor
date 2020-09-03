import unittest
import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint

from pyrotor.linear_conditions import get_endpoints_matrix
from pyrotor.linear_conditions import get_endpoints_bounds
from pyrotor.linear_conditions import get_implicit_matrix
from pyrotor.linear_conditions import get_implicit_bounds
from pyrotor.linear_conditions import get_linear_conditions


def test_get_endpoints_matrix():
    basis_dimensions = {"A": 2}
    endpoints = {"A": {"start": 1, "end": 2, "delta": 1}}
    expected = np.array([[1., -1.],
                         [1.,  1.]])
    endpoints_matrix = get_endpoints_matrix(basis_dimensions, endpoints)
    assert (expected == endpoints_matrix).all()

    basis_dimension = {"A": 3, "B": 2}
    endpoints = {'B': {'start': 5, 'end': 7, 'delta': 2},
                 'A': {'start': 1, 'end': 2, 'delta': 1}}
    expected = np.array([[1., -1.,  1.,  0.,  0.],
                         [0.,  0.,  0.,  1., -1.],
                         [1.,  1.,  1.,  0.,  0.],
                         [0.,  0.,  0.,  1.,  1.]])
    endpoints_matrix = get_endpoints_matrix(basis_dimension, endpoints)
    assert (expected == endpoints_matrix).all()

    basis_dimensions = {"A": 3, "B": 2}
    endpoints = {"B": {"start": 1, "end": 2, "delta": 1}}

    expected_phi = np.array([[0, 0,  0,  1,  -1.],
                             [0,  0,  0,  1., 1.]])

    phi = get_endpoints_matrix(basis_dimensions, endpoints)
    assert (phi == expected_phi).all()


def test_get_implicit_matrix():
    lambda_sigma = np.diag([0., 0., 0., 0., 1., 3.])
    x = 1 / np.sqrt(2)
    v = np.array([[0., x, 0., 0., x, 0.],
                  [1., 0., 0., 0., 0., 0.],
                  [0., x, 0., 0., -x, 0.],
                  [0., 0., x, 0., 0., -x],
                  [0., 0., 0., 1., 0., 0.],
                  [0., 0., x, 0., 0., x]])
    phi = np.dot(np.array([[1, 0., 0., 0., 0., 0.],
                           [0., np.sqrt(2), 0., 0., 0., 0.]]), v.T)
    sigma = np.linalg.multi_dot([v, lambda_sigma, v.T])

    null_space_sigma_phi = get_implicit_matrix(sigma, phi)
    expected_null_space_sigma_phi = np.array([v[:,3], -v[:,2]]).T

    np.testing.assert_almost_equal(null_space_sigma_phi, expected_null_space_sigma_phi)


def test_get_implicit_bounds():
    coef1 = np.array([1., 2., 3., 4., 5., 6.])
    coef2 = np.array([2., 3., 4., 5., 6., 7.])
    coef3 = np.array([0., 1., 2., 3., 4., 5.])
    coefficients = [coef1, coef2, coef3]
    null_space_sigma_phi = np.array([[1., 0.],
                                     [0., 0.],
                                     [0., 0.],
                                     [0, 0.],
                                     [0., 1.],
                                     [0, 0.]])

    projected_mean_coefficients = get_implicit_bounds(null_space_sigma_phi, coefficients)
    expected_projected_mean_coefficients = np.array([1., 5.])

    np.testing.assert_equal(projected_mean_coefficients, expected_projected_mean_coefficients)


def test_get_endpoints_bounds():
    variables = ['A', 'B']
    endpoints = {'B': {'start': 5, 'end': 7, 'delta': 2},
                 'A': {'start': 1, 'end': 2, 'delta': 1}}

    left_endpoints, right_endpoints = get_endpoints_bounds(endpoints, variables)
    expected_left_endpoints = [0, 3, 1, 5]
    expected_right_endpoints = [2, 7, 3, 9]

    assert left_endpoints == expected_left_endpoints
    assert right_endpoints == expected_right_endpoints

    variables = ['A', 'B']
    endpoints = {'B': {'start': 5, 'end': 7, 'delta': 2}}
    left_endpoints, right_endpoints = get_endpoints_bounds(endpoints, variables)
    expected_left_endpoints = [3, 5]
    expected_right_endpoints = [7, 9]

    assert left_endpoints == expected_left_endpoints
    assert right_endpoints == expected_right_endpoints


def test_get_linear_conditions():
    basis_dimensions = {"A": 3, "B": 3}
    endpoints = {'B': {'start': 5, 'end': 7, 'delta': 2},
                 'A': {'start': 1, 'end': 2, 'delta': 1}}
    coef1 = np.array([1., 2., 3., 4., 5., 6.])
    coef2 = np.array([2., 3., 4., 5., 6., 7.])
    coef3 = np.array([0., 1., 2., 3., 4., 5.])
    coefficients = [coef1, coef2, coef3]
    lambda_sigma = np.diag([0., 0., 0., 0., 1., 3.])
    x = 1 / np.sqrt(2)
    v = np.array([[0., x, 0., 0., x, 0.],
                  [1., 0., 0., 0., 0., 0.],
                  [0., x, 0., 0., -x, 0.],
                  [0., 0., x, 0., 0., -x],
                  [0., 0., 0., 1., 0., 0.],
                  [0., 0., x, 0., 0., x]])
    sigma = np.linalg.multi_dot([v, lambda_sigma, v.T])

    linear_conditions, linear_conditions_matrix = get_linear_conditions(basis_dimensions, endpoints, coefficients, sigma)
    expected_left_endpoints = [0., 3., 1., 5.]
    expected_right_endpoints = [2, 7, 3, 9]
    expected_linear_conditions_matrix = np.array([[1., -1.,  1.,  0.,  0., 0.],
                                                  [0.,  0.,  0.,  1., -1., 1.],
                                                  [1.,  1.,  1.,  0.,  0., 0.],
                                                  [0.,  0.,  0.,  1.,  1., 1.]])

    np.testing.assert_almost_equal(expected_linear_conditions_matrix, linear_conditions.A)
    np.testing.assert_almost_equal(expected_left_endpoints, linear_conditions.lb)
    np.testing.assert_almost_equal(expected_right_endpoints, linear_conditions.ub)
    np.testing.assert_almost_equal(expected_linear_conditions_matrix, linear_conditions_matrix)
