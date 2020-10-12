# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Project trajectories into a discrete format.
"""

from multiprocessing import Pool

import numpy as np
import pandas as pd

from numpy.polynomial.legendre import Legendre


def trajectory_to_coef(y, basis, basis_dimension):
    """
    Given a trajectory, compute its associated coefficients for each
    state with respect to a functional basis.

    Inputs:
        - y: DataFrame
            Trajectory - Index has to start at 0
        - basis: string
            Name of the functional basis
        - basis_dimension: dict
            Give the number of basis functions for each state
    Output:
        - coef: list of pd.Series
            Each element of the list contains the coefficients of a
            state
    """
    evaluation_points_nb = y.shape[0] - 1
    coef = []
    # Compute coefficients for each state
    for state in basis_dimension:
        if basis == 'legendre':
            # NB: Use Legendre class to fix the domain of the basis
            # Here consider each trajectory to be defined on [0,1]
            least_square_fit = Legendre.fit(y.index / evaluation_points_nb,
                                            y[state],
                                            deg=basis_dimension[state]-1,
                                            domain=[0, 1])
            s = pd.Series(least_square_fit.coef, name=state)
            coef.append(s)
    coef = np.array([c for series in coef for c in series.values])

    return coef


def trajectories_to_coefs(trajectories, basis, basis_dimension, n_jobs):
    """
    Given trajectories, compute their associated coefficients for each
    state with respect to a functional basis.

    Inputs:
        - y: list of DataFrame
            List of trajectories - Index has to start at 0
        - basis: string
            Functional basis
        - basis_dimension: dict
            Give the number of basis functions for each state
        - n_jobs: int
            Number of processes to use - If None, no parallelization is
            used
    Output:
        - coefs: list of pd.Series
            Each element of the list contains coefficients of a
            trajectory
    """
    if n_jobs is None:
        coefs = []
        for trajectory in trajectories:
            coef_i = trajectory_to_coef(trajectory, basis, basis_dimension)
            coefs.append(coef_i)
    else:
        # Format into iterable arguments
        basis = [basis for trajectory in trajectories]
        basis_dimension = [basis_dimension for trajectory in trajectories]
        args = zip(trajectories, basis, basis_dimension)
        # Creating a pool of workers
        with Pool(n_jobs) as pool:
            coefs = pool.starmap(trajectory_to_coef, args)

    return coefs


def compute_weighted_coef(coefs, weights, basis_dimension):
    """
    Compute weighted sum of trajectories through coefficients.

    Inputs:
        - coefs: list of pd.Series
            Each element of the list contains coefficients of a
            trajectory
        - weights: ndarray
            Vector containing the weights
        - basis_dimension: dict
            Give the number of basis functions for each state
    Output:
        c_weight: ndarray
            Vector containing weighted sum of the coefficients
    """
    K = sum(basis_dimension.values())
    c_weight = np.zeros(K)
    for i, coef_i in enumerate(coefs):
        c_weight += coef_i * weights[i]

    return np.array(c_weight)


def coef_to_trajectory(c, evaluation_points_nb, basis, basis_dimension):
    """
    Given coefficients, build the associated trajectory with respect to
    a functional basis.

    Inputs:
        - c: list of floats or list of pd.Series
            Each element of the list contains coefficients of a state
        - evaluation_points_nb: int
            Number of points on which the trajectory is evaluated
        - basis: string
            Name of the functional basis
        - basis_dimension: dict
            Give the number of basis functions for each state
    Output:
        - y: DataFrame
            Contains computed states of a flight
    """
    # FIXME: if below necessary ??
    # If c is list of floats, convert it into a list of pd.Series
    n_var = len(basis_dimension)
    if len(c) != n_var:
        c_formatted = []
        k = 0
        for state in basis_dimension:
            c_ = pd.Series(c[k:k+basis_dimension[state]], name=state)
            k += basis_dimension[state]
            c_formatted.append(c_)
        c = c_formatted.copy()

    y = pd.DataFrame()
    # Build each state
    for i in range(n_var):
        if basis == 'legendre':
            # Fix the domain [0,1] of the basis
            cl_c_state = Legendre(c[i].values, domain=[0, 1])
            # Evaluate
            _, y[c[i].name] = Legendre.linspace(cl_c_state,
                                                n=evaluation_points_nb)

    return y
