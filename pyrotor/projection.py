# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Project trajectories into a discrete format
"""

import numpy as np
import pandas as pd

from numpy.polynomial.legendre import Legendre


def trajectory_to_coef(y, basis, basis_dimension):
    """
    Given a trajectory, compute its associated coefficients for each variable
    with respect to a functional basis

    Inputs:
        - y: DataFrame
            Trajectory - Index has to start at 0
        - basis: string
            Name of the functional basis
        - basis_dimension: dict
            Give the number of basis functions for each variable

    Output:
        - coef: list of pd.Series
            Each element of the list contains the coefficients of a variable
    """
    # Number of evaluation points points
    evaluation_points_nb = y.shape[0] - 1
    coef = []
    # Compute coefficients for each variable
    for variable in basis_dimension:
        if basis == 'legendre':
            # NB: Use Legendre class to fix the domain of the basis
            # Here consider each trajectory to be defined on [0,1]
            least_square_fit = Legendre.fit(y.index / evaluation_points_nb,
                                            y[variable],
                                            deg=basis_dimension[variable]-1,
                                            domain=[0, 1])
            s = pd.Series(least_square_fit.coef, name=variable)
            coef.append(s)

    return coef


def trajectories_to_coefs(y, basis, basis_dimension):
    """
    Given trajectories, compute their associated coefficients for each variable
    with respect to a functional basis

    Inputs:
        - y: list of DataFrame
            List of trajectories - Index has to start
            at 0
        - basis: string
            Functional basis
        - basis_dimension: dict
            Give the number of basis functions for each variable

    Output:
        - coefs: list of pd.Series
            Each element of the list contains coefficients of a trajectory
    """
    coefs = []
    for _, y_i in enumerate(y):
        # Compute the coefficient of each flight
        coef_i = trajectory_to_coef(y_i, basis, basis_dimension)
        # Format into a numpy array
        coef_i = np.array([c for series in coef_i for c in series.values])
        coefs.append(coef_i)

    return coefs


def compute_weighted_coef(coefs, weights, basis_dimension):
    """
    Compute weighted sum of trajectories through coefficients

    Inputs:
        - coefs: list of pd.Series
            Each element of the list contains coefficients of a trajectory
        - weights: ndarray
            Vector containing the weights
        - basis_dimension: dict
            Give the number of basis functions for each variable

    Output:
        c_weight: ndarray
            Vector containing weighted sum of the coefficients
    """
    # Compute the dimension of the problem
    K = sum(basis_dimension.values())
    c_weight = np.zeros(K)
    # Compute the weighted sum
    for i, coef_i in enumerate(coefs):
        c_weight += coef_i * weights[i]

    return np.array(c_weight)


def coef_to_trajectory(c, evaluation_points_nb, basis, basis_dimension):
    """
    Given coefficients, build the associated trajectory with respect to a
    functional basis

    Inputs:
        - c: list of floats or list of pd.Series
            Each element of the list contains coefficients of a variable
        - evaluation_points_nb: int
            Number of points on which the trajectory is evaluated
        - basis: string
            Functionnal basis to project the flight on.
        - basis_dimension: dict
            Give the number of basis functions for each variable

    Output:
        - y: DataFrame
            Contains computed variables of a flight

    # FIXME: if below necessary ??
    """
    # If c is list of floats, convert it into a list of pd.Series
    # Compute number of variables
    n_var = len(basis_dimension)
    if len(c) != n_var:
        c_formatted = []
        for variable in basis_dimension:
            c_ = pd.Series(c[:basis_dimension[variable]], name=variable)
            del c[:basis_dimension[variable]]
            c_formatted.append(c_)
        c = c_formatted.copy()
    y = pd.DataFrame()
    # Build each variable
    for i in range(n_var):
        if basis == 'legendre':
            # Fix the domain [0,1] of the basis
            cl_c_variable = Legendre(c[i].values, domain=[0, 1])
            # Evaluate
            _, y[c[i].name] = Legendre.linspace(cl_c_variable,
                                                n=evaluation_points_nb)

    return y
