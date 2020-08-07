# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Compute the cost of a trajectory or directly on its coefficents
"""

import numpy as np


def compute_f(vector_x, sigma_inverse, c_weight):
    """
    Evaluate the coefficients of a single trajectory over f. Where f is the
    cost function given by the user.

    Inputs:
        - vector_x: ndarray
            Coefficients of a single trajectory.
        - sigma_inverse: ndarray
            Pseudoinverse of the covariance matrix of the reference
            coefficients.
        - c_weight: ndarray
            Coefficients of a weighted trajectory

    Output:
        - cost: float
            The cost of the given trajectory (by its coefficients) over the
            cost function given by the user.
    """
    part_a = np.dot(np.dot(vector_x.T, sigma_inverse), vector_x)
    part_b = np.dot(sigma_inverse, c_weight).T
    part_c = np.dot(2 * part_b, vector_x)
    return part_a - part_c


def compute_g(vector_x, matrix_q, vector_w):
    """
    Evaluate the coefficients of a single trajectory over g. Where g is the
    function penalizing the distance between the optimized trajectory and the
    reference trajectories.

    Inputs:
        - vector_x: ndarray
            Coefficients of a single trajectory.
        - matrix_q: ndarray
            Matrix of the quadratic term.
        - vector_w: ndarray
            Vector of the linear term (without intercept).

    Output:
        - g(vector_x): float
            Evaluation of vector_x over g.
    """
    part_a = np.dot(np.dot(vector_x.T, matrix_q), vector_x)
    part_b = np.dot(vector_w.T, vector_x)
    return part_a + part_b


def compute_cost(trajectory, quadratic_model):
    """
    Compute the cost of a trajectory given the quadratic model of the user.
    It is a vectorized version.

    Inputs:
        - trajectory: DataFrame
            Your trajecotry
        - quadratic_model: tuple or list
             The quadratic model of your cost function.
             Ex: (np.array([[1, 0], [2, 3]]), np.array([2, 1]), 8)
    """
    constant_part = quadratic_model[0]
    linear_part = quadratic_model[1]
    quadratic_part = quadratic_model[2]

    constant_result = constant_part * trajectory.shape[0]

    linear_result = np.sum(trajectory.values * linear_part)

    quadratic_result = np.dot(trajectory.values, quadratic_part)
    # ref: https://stackoverflow.com/questions/14758283/is-there-a-numpy-scipy-
    # dot-product-calculating-only-the-diagonal-entries-of-the
    quadratic_result = (quadratic_result * trajectory.values).sum(-1)
    quadratic_result = np.sum(quadratic_result)

    return constant_result + linear_result + quadratic_result


def compute_trajectories_cost(trajectories, quadratic_model,
                              basis_dimension=None):
    """
    Compute the cost for each trajectory of a list

    Inputs:
        - trajectories: list of pd.DataFrame
            Each element of the list is a trajectory
        - quadratic_model: str or list
            if str then it is the path to the folder containing the pickle
            model; else the first element of the list is w and the second one
            is q
        - basis_dimension: dict, default=None
            Give the number of basis functions for each variable

    Output:
        - trajectories_cost: ndarray
            Array containing the cost of the trajectories
    """
    # If pickle model, compute w, q using model_to_matrix()
    if isinstance(quadratic_model, str):
        # Compute w, q associated with the quadratic model
        vector_w, matrix_q = model_to_matrix(quadratic_model, basis_dimension)
    # Else extract w, q from quad_model
    else:
        vector_w, matrix_q = quadratic_model[0], quadratic_model[1]
    trajectories_cost = []
    # For each trajectory, compute total cost
    for trajectory in trajectories:
        trajectory_cost = compute_cost(trajectory, quadratic_model)
        trajectories_cost.append(trajectory_cost)

    return np.array(trajectories_cost)