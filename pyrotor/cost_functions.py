# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Compute the cost of a trajectory or directly via its coefficents
"""

from sklearn.pipeline import Pipeline
import numpy as np
import pickle


def compute_f(vector_x, sigma_inverse, c_weight):
    """
    Evaluate the coefficients of a single trajectory over g, where g is the
    function penalizing the distance between the optimized trajectory and the
    reference trajectories

    Inputs:
        - vector_x: ndarray
            Coefficients of a single trajectory
        - sigma_inverse: ndarray
            Pseudoinverse of the covariance matrix of the reference
            coefficients.
        - c_weight: ndarray
            Coefficients of a weighted trajectory

    Output:
        - cost: float
            The cost of the given trajectory (by its coefficients) over the
            cost function given by the user
    """
    part_a = np.dot(np.dot(vector_x.T, sigma_inverse), vector_x)
    part_b = np.dot(sigma_inverse, c_weight).T
    part_c = np.dot(2 * part_b, vector_x)
    return part_a - part_c


def compute_g(vector_x, matrix_q, vector_w):
    """
    Evaluate the coefficients of a single trajectory over f, where f is the
    cost function given by the user

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


def predict_cost_by_time(trajectory, quadratic_model):
    """

    """
    # Compute cost directly from the sklearn model
    cost_by_time = quadratic_model.predict(trajectory.values)
    return cost_by_time


def compute_cost_by_time(trajectory, quadratic_model):
    """
    Compute the cost of a trajectory at every time point given the quadratic
    model of the user - Vectorized version

    Inputs:
        - trajectory: DataFrame
            Your trajecotry
        - quadratic_model: tuple or list
             The quadratic model of your cost function.
             Ex: (np.array([[1, 0], [2, 3]]), np.array([2, 1]), 8)

    Output:
        - cost_by_time: ndarray
            The trajectory cost at each time point
    """
    if isinstance(quadratic_model, Pipeline):
        return predict_cost_by_time(trajectory, quadratic_model)
    else:
        # FIXME: verify computations
        constant_part = quadratic_model[0]
        linear_part = quadratic_model[1]
        quadratic_part = quadratic_model[2]
        trajectory = trajectory.values
        constant_costs = constant_part * np.ones(trajectory.shape[0], dtype=np.float32)
        # TODO: include sampling frequency ?
        linear_costs = np.sum(trajectory * linear_part, axis=1)

        quadratic_costs = np.dot(trajectory, quadratic_part)
        # ref: https://stackoverflow.com/questions/14758283/is-there-a-numpy-scipy-
        # dot-product-calculating-only-the-diagonal-entries-of-the
        quadratic_costs = (quadratic_costs * trajectory).sum(-1)

        return constant_costs + linear_costs + quadratic_costs


def compute_cost(trajectory, quadratic_model):
    """
    Compute the cost of a trajectory given the quadratic model of the user -
    Vectorized version

    Inputs:
        - trajectory: DataFrame
            Your trajecotry
        - quadratic_model: tuple or list
             The quadratic model of your cost function.
             Ex: (np.array([[1, 0], [2, 3]]), np.array([2, 1]), 8)

    Output:
        - trajectory_cost: float
            The total cost of the trajectory
    """
    if isinstance(quadratic_model, Pipeline):
        cost_by_time = predict_cost_by_time(trajectory,
                                            quadratic_model)
    else:
        cost_by_time = compute_cost_by_time(trajectory,
                                            quadratic_model)
    return np.sum(cost_by_time)


def load_model(name):
    """
    Load model saved in a .pkl format

    Input:
        - name: string
            Name of the .pkl file

    Output:
        - model: Python object
            Loaded machine learning model, scipy class...
    """
    with open(name, 'rb') as file:
        model = pickle.load(file)

    return model


def compute_trajectories_cost(trajectories, quadratic_model):
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
            Give the number of basis functions for each state

    Output:
        - trajectories_cost: ndarray
            Array containing the cost of the trajectories
    """
    # If pickle model, compute w, q using model_to_matrix()
    trajectories_cost = []
    for trajectory in trajectories:
        trajectory_cost = compute_cost(trajectory, quadratic_model)
        trajectories_cost.append(trajectory_cost)
    return np.array(trajectories_cost)
