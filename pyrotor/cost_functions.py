# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Compute the cost of a trajectory or directly on its coefficents
"""

import numpy as np
import pickle


def compute_f(vector_x, sigma_inverse, c_weight):
    """
    Evaluate the coefficients of a single trajectory over g. Where g is the
    function penalizing the distance between the optimized trajectory and the
    reference trajectories.

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
    Evaluate the coefficients of a single trajectory over f. Where f is the
    cost function given by the user.

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


def compute_cost_by_time(trajectory, quadratic_model):
    """
    Compute the cost of a trajectory at every time point given the quadratic
    model of the user.
    It is a vectorized version.

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
    constant_part = quadratic_model[0]
    linear_part = quadratic_model[1]
    quadratic_part = quadratic_model[2]

    trajectory = trajectory.values

    constant_costs = constant_part * np.ones(trajectory.shape[0])
    # to do include sampling frequency
    linear_costs = np.sum(trajectory * linear_part, axis=1)

    quadratic_costs = np.dot(trajectory, quadratic_part)
    # ref: https://stackoverflow.com/questions/14758283/is-there-a-numpy-scipy-
    # dot-product-calculating-only-the-diagonal-entries-of-the
    quadratic_costs = (quadratic_costs * trajectory).sum(-1)
    quadratic_costs = quadratic_costs

    print(linear_costs)

    return constant_costs + linear_costs + quadratic_costs


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

    Output:
        - trajectory_cost: float
            The total cost of the trajectory
    """
    cost_by_time = compute_cost_by_time(trajectory, quadratic_model)
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


def model_to_matrix(path, n_var):
    """
    From a quadratic model f(x), compute q and w such that
    f(x) = <x, qx> + <w, x>

    Inputs:
        - path: string
            Path for the folder containing the pickle model for FF
        - n_var: int
            Give the number of variable

    Outputs:
        - w: numpy array [1, n_var]
            Vector of the linear term (without intercept)
        - q: numpy arry [n_var, n_var]
            Matrix of the quadratic term
    """
    # Load model
    model = load_model(path)
    # Get coefficients of the model (from make_pipeline of sk-learn)
    coef = np.array(model.named_steps['lin_regr'].coef_)
    # Remove normalization from StandardScaler()
    std_ = np.sqrt(model.named_steps['scale'].var_)
    coef /= std_
    # Add the constant
    c = model.named_steps['lin_regr'].intercept_
    # Define w
    w = coef[1:n_var+1]
    # Define q starting by the upper part and deduce then the lower one
    coef = np.delete(coef, range(n_var+1))
    # Divide coef by two because a x^2 + b xy + c y^2 is associated with
    # [[a, b/2],[b/2, c]]
    coef /= 2
    q = np.zeros([n_var, n_var])
    for i in range(n_var):
        q[i, i:] += coef[:n_var - i]
        # Mutliply the diagonal by 2
        q[i, i] *= 2
        coef = np.delete(coef, range(n_var - i))
    # Deduce the lower part
    q += np.transpose(np.triu(q, 1))

    return (c, w, q)


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
            Give the number of basis functions for each variable

    Output:
        - trajectories_cost: ndarray
            Array containing the cost of the trajectories
    """
    # If pickle model, compute w, q using model_to_matrix()
    trajectories_cost = []
    if isinstance(quadratic_model, str):
        # Compute cost directly from the sklearn model
        model = load_model(quadratic_model)
        for trajectory in trajectories:
            trajectory_cost = np.sum(model.predict(trajectory.values))
            trajectories_cost.append(trajectory_cost)
    else:
        # For each trajectory, compute total cost from the algebra formulation
        for trajectory in trajectories:
            trajectory_cost = compute_cost(trajectory,
                                           quadratic_model)
            trajectories_cost.append(trajectory_cost)
    print(len(trajectories_cost))
    return np.array(trajectories_cost)
