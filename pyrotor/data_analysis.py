# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
File to analyze the reference trajectories
"""

import numpy as np

from sklearn.covariance import EmpiricalCovariance
from sklearn.covariance import GraphicalLasso

from .objective_matrices import model_to_matrix


def nb_samples_is_sufficient(X):
    if np.shape(X)[0] > 2 * np.shape(X)[1]:
        return True
    else:
        return False


def compute_covariance(X):
    """
    Estimate covariance and precision matrices from data X - Depending on
    samples number, use either EmpiricalCovariance or GraphicalLasso methods
    from scikit-learn

    Input:
        X: ndarray
            Data

    Outputs:
        covariance: ndarray
            Estimated covariance matrix
        precision: ndarray
            Estimated precision matrix (i.e. pseudo-inverse of covariance)
    """
    if nb_samples_is_sufficient(X):
        cov = EmpiricalCovariance().fit(X)
    else:
        cov = GraphicalLasso(mode='lars').fit(X)
    covariance = cov.covariance_
    precision = cov.precision_

    return covariance, np.diag(np.diag(precision))


def compute_trajectories_cost(trajectories, quad_model, basis_dimension=None):
    """
    Compute the cost for each trajectory of a list

    Inputs:
        - trajectories: list of pd.DataFrame
            Each element of the list is a trajectory
        - quad_model: str or list
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
    if isinstance(quad_model, str):
        # Compute w, q associated with the quadratic model
        w, q = model_to_matrix(quad_model, basis_dimension)
    # Else extract w, q from quad_model
    else:
        w, q = quad_model[0], quad_model[1]
    trajectories_cost = []
    # For each trajectory, compute total cost
    for trajectory in trajectories:
        x = trajectory.values.T
        points_nb = x.shape[1]
        # Compute quadratic term value for each observation
        quadratic_term = [np.linalg.multi_dot([x[:,m].T, q, x[:,m]])
                          for m in range(points_nb)]
        # Compute linear term value for each observation
        linear_term = np.dot(w, x)
        trajectory_pointwise_cost = quadratic_term + linear_term
        # Compute total cost by addition
        trajectories_cost.append(np.sum(trajectory_pointwise_cost))

    return np.array(trajectories_cost)


def select_trajectories(trajectories, trajectories_cost, trajectories_nb):
    """
    Return the trajectories associated with the smallest costs

    Inputs:
        - trajectories: list of pd.DataFrame
            Each element of the list is a trajectory
        - trajectories_cost: ndarray
            Array containing the cost of the trajectories
        - trajectories_nb: int
            Number of trajectories to keep

    Ouputs:
        - best_trajectories: list of pd.DataFrame
            List containing the best trajectories

    """
    # Sort indices with respect to costs
    I = sorted(range(len(trajectories)), key = lambda k: trajectories_cost[k])
    # Keep the first ones
    best_trajectories = [trajectories[i] for i in I[:trajectories_nb]]

    return best_trajectories


def compute_weights(trajectories_cost, f=None):
    """
    Compute normalized weights associated with each trajectory

    Inputs:
        - trajectories_cost: ndarray
            Array containing the cost of the trajectories
        - f: function, default=None
            Function used to compute weights from the costs - Default function
            is f(x) = exp(-x)
    """
    if f:
        weights = f(trajectories_cost)
    else:
        weights = np.exp(-trajectories_cost)
    # Normalize
    weights = weights / np.sum(weights)

    return weights


def compute_intersection_kernels(A, B):
    """
    """
