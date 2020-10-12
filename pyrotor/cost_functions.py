# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Compute the cost of a trajectory.
"""

from sklearn.pipeline import Pipeline
import numpy as np
import pickle


def load_model(name):
    """
    Load model saved in a .pkl format.

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


def compute_f(vector_x, sigma_inverse, c_weight):
    """
    Compute cost of a trajectory through its coefficients.

    Inputs:
        - vector_x: ndarray
            Coefficients of a single trajectory
        - sigma_inverse: ndarray
            Pseudoinverse of the covariance matrix of the reference
            coefficients
        - c_weight: ndarray
            Coefficients of a weighted trajectory
    Output:
        - cost: float
            The cost of the given trajectory
    """
    part_a = np.dot(np.dot(vector_x.T, sigma_inverse), vector_x)
    part_b = np.dot(sigma_inverse, c_weight).T
    part_c = np.dot(2 * part_b, vector_x)

    return part_a - part_c


def compute_g(vector_x, matrix_q, vector_w):
    """
    Compute the distance mean of a trajectory to the reference ones
    through its coefficients.

    Inputs:
        - vector_x: ndarray
            Coefficients of a single trajectory
        - matrix_q: ndarray
            Matrix of the quadratic term
        - vector_w: ndarray
            Vector of the linear term (without intercept)
    Output:
        - g(vector_x): float
            Distance of the trajectory to the reference ones
    """
    part_a = np.dot(np.dot(vector_x.T, matrix_q), vector_x)
    part_b = np.dot(vector_w.T, vector_x)

    return part_a + part_b


def predict_cost_by_time(trajectory, quadratic_model):
    """
    Predict the cost for each time of the trajectory from a quadratic
    trained model.

    Inputs:
        trajectory: ndarray
            Trajectory of interest
        quadratic_model: sklearn.pipeline.Pipeline object
            Quadratic model modelling trajectory cost
    Output:
        cost_by_time: ndarray
            Predicted cost by time of your trajectory
    """
    # Compute cost directly from the sklearn model
    cost_by_time = quadratic_model.predict(trajectory.values)

    return cost_by_time


def compute_cost_by_time(trajectory, quadratic_model):
    """
    Compute the cost of a trajectory at every time given a quadratic
    model - Vectorized version.

    Inputs:
        - trajectory: DataFrame
            Trajectory of interest
        - quadratic_model: str or list
            if str then it is the path to the folder containing the
            pickle model; else the first element of the list is w,
            the second one is q and the third is the constant
            Ex: (np.array([[1, 0], [2, 3]]), np.array([2, 1]), 8)
    Output:
        - cost_by_time: ndarray
            Trajectory cost at each time
    """
    if isinstance(quadratic_model, Pipeline):
        return predict_cost_by_time(trajectory, quadratic_model)
    else:
        constant_part = quadratic_model[0]
        linear_part = quadratic_model[1]
        quadratic_part = quadratic_model[2]
        trajectory = trajectory.values
        constant_costs = constant_part * np.ones(trajectory.shape[0],
                                                 dtype=np.float32)
        # TODO: include sampling frequency ?
        linear_costs = np.sum(trajectory * linear_part, axis=1)
        # ref: https://stackoverflow.com/questions/14758283/is-there-a
        # -numpy-scipy-dot-product-calculating-only-the-diagonal-entries
        # -of-the
        quadratic_costs = np.dot(trajectory, quadratic_part)
        quadratic_costs = (quadratic_costs * trajectory).sum(-1)

        return constant_costs + linear_costs + quadratic_costs


def compute_cost(trajectory, quadratic_model):
    """
    Compute total cost of a trajectory given a quadratic model -
    Vectorized version.

    Inputs:
        - trajectory: DataFrame
            Trajectory of interest
        - quadratic_model: str or list
            if str then it is the path to the folder containing the
            pickle model; else the first element of the list is w,
            the second one is q and the third is the constant
            Ex: (np.array([[1, 0], [2, 3]]), np.array([2, 1]), 8)
    Output:
        - trajectory_cost: float
            Total cost of the trajectory
    """
    if isinstance(quadratic_model, Pipeline):
        cost_by_time = predict_cost_by_time(trajectory,
                                            quadratic_model)
    else:
        cost_by_time = compute_cost_by_time(trajectory,
                                            quadratic_model)

    return np.sum(cost_by_time)


def compute_trajectories_cost(trajectories, quadratic_model):
    """
    Compute the cost for each trajectory of a list.

    Inputs:
        - trajectories: list of pd.DataFrame
            Each element of the list is a trajectory
        - quadratic_model: str or list
            if str then it is the path to the folder containing the
            pickle model; else the first element of the list is w,
            the second one is q and the third is the constant
            Ex: (np.array([[1, 0], [2, 3]]), np.array([2, 1]), 8)
    Output:
        - trajectories_cost: ndarray
            Array containing the cost of the trajectories
    """
    trajectories_cost = []
    for trajectory in trajectories:
        trajectory_cost = compute_cost(trajectory, quadratic_model)
        trajectories_cost.append(trajectory_cost)

    return np.array(trajectories_cost)
