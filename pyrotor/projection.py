# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Transform trajectories into a discrete format
"""

import numpy as np
import pandas as pd
from numpy.polynomial import legendre


def trajectory_to_coef(y, duration, basis, var):
    """
    Given states for one flight, compute coefficients of each state on a given
    functional basis

    Inputs:
        - y: DataFrame
            Contains recorded states of a flight - Index has to start at 0
        - duration: int
            Duration of the flight
        - basis: string
            Functional basis
        - var: dict
            Give the number of basis functions for each state

    Output:
        - c: list of pd.Series
            Each element of the list contains coefficients of a state
    """
    # Length of the time interval
    T = duration
    c = []
    # Compute coefficients for each state
    for state in y.columns:
        if basis == 'legendre':
            # NB: Use Legendre class to fix the domain of the basis
            # Here consider each flight as defined on [0,1]
            cl_c_state = legendre.Legendre.fit(y.index / T, y[state],
                                            deg=var[state]-1, domain=[0, 1])
            s = pd.Series(cl_c_state.coef, name=state)
            c.append(s)

    return c


def trajectories_to_coefs(y, durations, basis, var):
    """
    Given states for several flights, compute coefficients of each state of
    each flight on a given functional basis

    Inputs:
        - y: list of DataFrame
            Contains recorded states of several flight - Index has to start
            at 0
        - durations: list of int
            Duration of each flight
        - basis: string
            Functional basis
        - var: dict
            Give the number of basis functions for each state

    Output:
        - c: list of pd.Series
            Each element of the list contains coefficients of a flight
    """

    # Init the list of coefficients
    coefs = []
    for i, y_i in enumerate(y):
        # Compute the coefficient of each flight
        coef_i = traj_to_coef(y_i, durations[i], basis, var)
        # Format into a numpy array
        coef_i = np.array([c for series in coef_i for c in series.values])
        coefs.append(coef_i)

    return coefs


def coef_to_trajectory(c, duration, basis, var):
    """
    Given coefficients for one flight, build each state

    Inputs:
        - c: list of floats or list of pd.Series
            Each element of the list contains coefficients of a state
        - duration: int
            Flight duration
        - basis: string
            Functionnal basis to project the flight on.
        - var: dict
            Give the number of basis functions for each state

    Output:
        - y: DataFrame
            Contains computed states of a flight
    """
    # If c is list of floats, convert it into a list of pd.Series
    # Compute number of variables
    n_var = len(var)
    if len(c) != n_var:
        c_formatted = []
        for state in var:
            c_ = pd.Series(c[:var[state]], name=state)
            del c[:var[state]]
            c_formatted.append(c_)
        c = c_formatted.copy()
    # Length of the time interval
    T = duration
    y = pd.DataFrame()
    # Build each state
    for i in range(n_var):
        if basis == 'legendre':
            # Initiate Legendre class to fix the domain [0,1] of the basis
            cl_c_state = legendre.Legendre(c[i].values, domain=[0, 1])
            # Evaluate on T points
            _, y[c[i].name] = legendre.Legendre.linspace(cl_c_state, n=T)

    return y


def integrate_bases_legendre(duration, var):
    """
    Compute vector containing mean for each element of the n_var bases and
    matrix containing dot products between each element of the n_var bases
    - Only for Legendre polynomials

    Inputs:
        - duration: int
            Flight duration
        - var: dict
            Give the number of basis functions for each state

    outputs:
        - mean: numpy array [1, d]
            Array containing the means over an interval
        - dot_product: numpy array [d, d]
            Matrix containing the dot products
    """
    # Length of the time interval
    # T = 1 because trajectory projected onto [0,1]
    T = 1
    # Compute the dimension of the problem
    d = np.sum([var[elt] for elt in var])
    # For Legendre polynomials, mean = 0 except when n=0
    mean = np.zeros(d)
    k = 0
    for state in var:
        mean[k] += T
        k += var[state]
    # Compute dot product between the d polynomials
    # Here use <P_n, P_m> = T / (2*n + 1) * delta_mn
    dot_product = np.zeros([d, d])
    k, l = 0, 0
    for state1 in var:
        for state2 in var:
            m = min([var[state1], var[state2]])
            for n in range(m):
                # Squared L^2-norm of the n-th Legendre polynomial
                dot_product[k + n, l + n] = T / (2*n + 1)
            l += var[state2]
        l = 0
        k += var[state1]

    return mean, dot_product
