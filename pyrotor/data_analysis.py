# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
File to analyze the reference trajectories
"""

import numpy as np

from sklearn.covariance import EmpiricalCovariance
from sklearn.covariance import GraphicalLasso

from .objective_matrices import model_to_matrix


def nb_samples_is_sufficient(dataset):
    """
    Tell wether or not you have enough samples in your data set.
    """
    if np.shape(dataset)[0] > 2 * np.shape(dataset)[1]:
        return True
    return False


def compute_covariance(dataset):
    """
    Estimate covariance and precision matrices from data X - Depending on
    samples number, use either EmpiricalCovariance or GraphicalLasso methods
    from scikit-learn

    Input:
        dataset: ndarray
            Dataset

    Outputs:
        covariance: ndarray
            Estimated covariance matrix
        precision: ndarray
            Estimated precision matrix (i.e. pseudo-inverse of covariance)
    """
    if nb_samples_is_sufficient(dataset):
        cov = EmpiricalCovariance().fit(dataset)
    else:
        cov = GraphicalLasso(mode='lars').fit(dataset)
    covariance = cov.covariance_
    precision = cov.precision_

    # return covariance, np.diag(np.diag(precision))
    return covariance, precision


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
    indexes = sorted(range(len(trajectories)),
                     key=lambda k: trajectories_cost[k])
    # Keep the first ones
    best_trajectories = [trajectories[i] for i in indexes[:trajectories_nb]]
    return best_trajectories


def compute_weights(trajectories_cost, weight_fonction=None):
    """
    Compute normalized weights associated with each trajectory

    Inputs:
        - trajectories_cost: ndarray
            Array containing the cost of the trajectories
        - weight_fonction: function, default=None
            Function used to compute weights from the costs - Default function
            is f(x) = exp(-x)
    """
    print(trajectories_cost)
    if weight_fonction:
        weights = weight_fonction(trajectories_cost)
    else:
        weights = np.exp(trajectories_cost-np.max(trajectories_cost))
    # Normalize
    weights = weights / np.sum(weights)

    return weights


def compute_intersection_kernels():
    """
    ompute the intersection kernel between A and B
    """
