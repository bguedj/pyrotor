# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
File to analyze the reference trajectories.
"""

import numpy as np

from sklearn.covariance import EmpiricalCovariance
from sklearn.covariance import GraphicalLasso

import warnings


def nb_samples_is_sufficient(dataset):
    """
    Tell wether or not there is enough samples in the data set.

    Input:
        dataset: ndarray
            Dataset
    Output:
        is_sufficient: boolean
            Tell wether or not there are twice more observations than
            features
    """
    if np.shape(dataset)[0] > 2 * np.shape(dataset)[1]:
        return True
    return False


def compute_covariance(dataset):
    """
    Estimate covariance and precision matrices from data X.
    
    Depending on samples number, use either EmpiricalCovariance or
    GraphicalLasso methods from scikit-learn.

    Input:
        dataset: ndarray
            Dataset
    Outputs:
        covariance: ndarray
            Estimated covariance matrix
        precision: ndarray
            Estimated precision matrix (i.e. pseudo-inverse of
            covariance)
    """
    # Turn matching warnings into exceptions
    warnings.filterwarnings("error")
    if nb_samples_is_sufficient(dataset):
        cov = EmpiricalCovariance().fit(dataset)
        covariance = cov.covariance_
        precision = cov.precision_
        return covariance, precision
    else:
        try:
            cov = GraphicalLasso(mode='cd').fit(dataset)
            covariance = cov.covariance_
            precision = cov.precision_
            return covariance, precision
        except Exception as e:
            lasso_error = str(e)
            raise ValueError(lasso_error
                             + '\nNumber of reference trajectories not '
                             'sufficiently large to estimate covariance '
                             'and precision matrices.')


def select_trajectories(trajectories, trajectories_cost, trajectories_nb):
    """
    Return the trajectories associated with the smallest costs.

    Inputs:
        - trajectories: list of pd.DataFrame
            Each element of the list is a trajectory
        - trajectories_cost: ndarray
            Array containing the cost of the trajectories
        - trajectories_nb: int
            Number of trajectories to keep
    Ouput:
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
    Compute normalized weights associated with each trajectory.

    Inputs:
        - trajectories_cost: ndarray
            Array containing the cost of the trajectories
        - weight_fonction: function, default=None
            Function used to compute weights from the costs - Default
            function is f(x) = exp(-x)
    Output:
        - weights: ndarray
            Array containing normalized weights
    """
    if weight_fonction:
        weights = weight_fonction(trajectories_cost)
    else:
        weights = np.exp(trajectories_cost-np.max(trajectories_cost))
    # Normalize
    weights = weights / np.sum(weights)

    return weights


def add_derivatives(reference_trajectories, basis_dimension):
    """
    Compute derivatives of each state from a dataframe and append
    derivatives to initial dataframe.

    Inputs:
        - reference_trajectories: list of DataFrame
            List of reference trajectories
        - basis_dimension: dict
            Give the number of basis functions for each state
    Output:
        reference_trajectories_deriv: list of DataFrame
            List of reference trajectories with derivatives
    """
    reference_trajectories_deriv = []
    for traj in reference_trajectories:
        traj_deriv = traj.copy(deep=True)
        # For each state, compute derivative
        for state in basis_dimension.keys():
            deriv_state = state + '_deriv'
            state_values = traj[state].values
            # Add a final derivative computed through the two last values
            state_fin = state_values[-2] \
                + 2 * (state_values[-1] - state_values[-2])
            # Compute derivative with np.diff
            traj_deriv[deriv_state] = np.diff(state_values, append=state_fin)
        reference_trajectories_deriv.append(traj_deriv)

    return reference_trajectories_deriv
