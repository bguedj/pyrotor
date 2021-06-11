# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Compute the cost of a trajectory.
"""

from sklearn.pipeline import Pipeline
import numpy as np
import pickle

from .projection import coef_to_trajectory

from scipy.integrate import trapz


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
    Compute the distance mean of a trajectory to the reference ones
    through its coefficients.

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
            Distance of the trajectory to the reference ones
    """
    part_a = np.dot(np.dot(vector_x.T, sigma_inverse), vector_x)
    part_b = np.dot(sigma_inverse, c_weight).T
    part_c = np.dot(2 * part_b, vector_x)

    return part_a - part_c


def compute_g(vector_x, format_model, extra_info):
    """
    Compute cost of a trajectory through its coefficients.

    Inputs:
        - vector_x: ndarray
            Coefficients of a single trajectory
        - format_model: list of arrays or sklearn model
            Model of the cost; if list, the first element of the list is the
            the integrated linear part W and the second one the integrated
            quadratic part Q
        - extra_info: dict
            Contains independent_variable, basis, basis_features and
            basis_dimension dictionaries
    Output:
        - g(vector_x): float
            The cost of the given trajectory
    """
    # If sklearn model, reconstruct trajectory and compute its cost via the
    # model
    if isinstance(format_model, Pipeline):
        # Get information to reconstruct the trajectory
        independent_variable = extra_info['independent_variable']
        basis = extra_info['basis']
        basis_features = extra_info['basis_features']
        basis_dimension = extra_info['basis_dimension']
        start = independent_variable['start']
        end = independent_variable['end']
        points_nb = independent_variable['points_nb']
        traj = coef_to_trajectory(vector_x, points_nb, basis, basis_features,
                                  basis_dimension)
        # Compute instantaneous cost
        cost_inst = format_model.predict(traj)
        # Compute total cost using trapezoids method
        g = trapz(cost_inst, np.linspace(start, end, points_nb))
    # Else use quadratic formula
    else:
        W, Q = format_model[0], format_model[1]
        # Compute quadratic part
        g = np.dot(np.dot(vector_x.T, Q), vector_x)
        # Compute linear part
        g += np.dot(W.T, vector_x)

    return g


def compute_cost_by_time(trajectory, model):
    """
    Compute the cost of a trajectory at every time given a quadratic
    model - Vectorized version.

    Inputs:
        - trajectory: DataFrame
            Trajectory of interest
        - model: list of arrays or sklearn model
            Model of the cost; if list, the first element of the list is the
            constant c, the second one is the linear part w and the third one
            is the quadratic part q
    Output:
        - cost_by_time: ndarray
            Trajectory cost at each time
    """
    # If sklearn model, compute instantaneous cost via the model
    if isinstance(model, Pipeline):
        return model.predict(trajectory.values)
    # Else use quadratic formula
    else:
        constant_part = model[0]
        linear_part = model[1]
        quadratic_part = model[2]
        trajectory = trajectory.values
        constant_costs = constant_part * np.ones(trajectory.shape[0],
                                                 dtype=np.float32)
        linear_costs = np.sum(trajectory * linear_part, axis=1)
        # ref: https://stackoverflow.com/questions/14758283/is-there-a
        # -numpy-scipy-dot-product-calculating-only-the-diagonal-entries
        # -of-the
        quadratic_costs = np.dot(trajectory, quadratic_part)
        quadratic_costs = (quadratic_costs * trajectory).sum(-1)

        return constant_costs + linear_costs + quadratic_costs


def compute_cost(trajectory, model, independent_variable):
    """
    Compute total cost of a trajectory given a quadratic model -
    Vectorized version.

    Inputs:
        - trajectory: DataFrame
            Trajectory of interest
        - model: list of arrays or sklearn model
            Model of the cost; if list, the first element of the list is the
            constant c, the second one is the linear part w and the third one
            is the quadratic part q
        - independent_variable: dict
            Describe the time-interval on which are defined the trajectories
            ex: {'start': 0, 'end': 1, 'frequency':.1}
    Output:
        - trajectory_cost: float
            Total cost of the trajectory
    """
    if isinstance(model, Pipeline):
        cost_by_time = model.predict(trajectory.values)
    else:
        cost_by_time = compute_cost_by_time(trajectory,
                                            model)
    start = independent_variable['start']
    end = independent_variable['end']
    points_nb = independent_variable['points_nb']

    return trapz(cost_by_time, np.linspace(start, end, points_nb))


def compute_trajectories_cost(trajectories, model, independent_variable):
    """
    Compute the cost for each trajectory of a list.

    Inputs:
        - trajectories: list of pd.DataFrame
            Each element of the list is a trajectory
        - model: list of arrays or sklearn model
            Model of the cost; if list, the first element of the list is the
            constant c, the second one is the linear part w and the third one
            is the quadratic part q
        - independent_variable: dict
            Describe the time-interval on which are defined the trajectories
            ex: {'start': 0, 'end': 1, 'frequency':.1}
    Output:
        - trajectories_cost: ndarray
            Array containing the cost of the trajectories
    """
    trajectories_cost = []
    for trajectory in trajectories:
        trajectory_cost = compute_cost(trajectory,
                                       model,
                                       independent_variable)
        trajectories_cost.append(trajectory_cost)

    return np.array(trajectories_cost)
