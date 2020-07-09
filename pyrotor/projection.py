# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Project trajectories into a discrete format
"""

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import Legendre


def trajectory_to_coef(y, basis, basis_dimensions):
    """
    Given variables for one flight, compute coefficients of each variable on a given
    functional basis

    Inputs:
        - y: DataFrame
            Contains recorded variables of a flight - Index has to start at 0
        - basis: string
            Functional basis
        - basis_dimensions: dict
            Give the number of basis functions for each variable

    Output:
        - c: list of pd.Series
            Each element of the list contains the coefficients of a variable
    """
    # Number of element
    n = y.shape[0]
    c = []
    # Compute coefficients for each variable
    for variable in y.columns:
        if basis == 'legendre':
            # NB: Use Legendre class to fix the domain of the basis
            # Here consider each trajectory to be defined on [0,1]
            least_square_fit = Legendre.fit(y.index / n,
                                            y[variable],
                                            deg=basis_dimensions[variable]-1,
                                            domain=[0, 1])
            s = pd.Series(least_square_fit.coef, name=variable)
            c.append(s)

    return c


def trajectories_to_coefs(y, basis, basis_dimensions):
    """
    Given variables for several flights, compute coefficients of each variable of
    each flight on a given functional basis

    Inputs:
        - y: list of DataFrame
            Contains recorded variables of several flight - Index has to start
            at 0
        - basis: string
            Functional basis
        - basis_dimensions: dict
            Give the number of basis functions for each variable

    Output:
        - c: list of pd.Series
            Each element of the list contains coefficients of a flight
    """

    # Init the list of coefficients
    coefs = []
    for i, y_i in enumerate(y):
        # Compute the coefficient of each flight
        coef_i = trajectory_to_coef(y_i, basis, basis_dimensions)
        # Format into a numpy array
        coef_i = np.array([c for series in coef_i for c in series.values])
        coefs.append(coef_i)

    return coefs


def coef_to_trajectory(c, time_range, basis, basis_dimensions):
    """
    Given coefficients for one flight, build each variable

    Inputs:
        - c: list of floats or list of pd.Series
            Each element of the list contains coefficients of a variable
        - duration: dict
            Information about the time domain (or independant variable).
            ex: {"t0": 1, "t1": 99, "dt": 0.1}
        - basis: string
            Functionnal basis to project the flight on.
        - basis_dimensions: dict
            Give the number of basis functions for each variable

    Output:
        - y: DataFrame
            Contains computed variables of a flight
    """
    t0 = time_range['t0']
    t1 = time_range['t1']
    dt = time_range['dt']
    duration = (t1 - t0)//dt
    # If c is list of floats, convert it into a list of pd.Series
    # Compute number of variables
    n_var = len(basis_dimensions)
    if len(c) != n_var:
        c_formatted = []
        for variable in basis_dimensions:
            c_ = pd.Series(c[:basis_dimensions[variable]], name=variable)
            del c[:basis_dimensions[variable]]
            c_formatted.append(c_)
        c = c_formatted.copy()
    # Length of the time interval
    y = pd.DataFrame()
    # Build each variable
    for i in range(n_var):
        if basis == 'legendre':
            # Initiate Legendre class to fix the domain [0,1] of the basis
            # FIXME: domain=[t0, t1]
            cl_c_variable = Legendre(c[i].values, domain=[t0, t1])
            # Evaluation on duration points
            _, y[c[i].name] = Legendre.linspace(cl_c_variable, n=duration)
    return y


def integrate_basis_legendre(basis_dimensions):
    """
    Compute a vector containing the mean of each element of the n_var bases and
    matrix containing dot products between each element of the n_var bases
    - Only for Legendre polynomials

    Inputs:
        - basis_dimensions: dict
            Give the number of basis functions for each variable

    Outputs:
        - mean: numpy array [1, d]
            Array containing the means over an interval
        - dot_product: numpy array [d, d]
            Matrix containing the dot products
    """
    # Length of the time interval
    # T = 1 because trajectory projected onto [0,1]
    T = 1
    # Compute the dimension of the problem
    d = np.sum([basis_dimensions[elt] for elt in basis_dimensions])
    # For Legendre polynomials, mean = 0 except when n=0
    mean = np.zeros(d)
    k = 0
    for variable in basis_dimensions:
        mean[k] += T
        k += basis_dimensions[variable]
    # Compute dot product between the d polynomials
    # Here use <P_n, P_m> = T / (2*n + 1) * delta_mn
    dot_product = np.zeros([d, d])
    k, l = 0, 0
    for variable1 in basis_dimensions:
        for variable2 in basis_dimensions:
            m = min([basis_dimensions[variable1], basis_dimensions[variable2]])
            for n in range(m):
                # Squared L^2-norm of the n-th Legendre polynomial
                dot_product[k + n, l + n] = T / (2*n + 1)
            l += basis_dimensions[variable2]
        l = 0
        k += basis_dimensions[variable1]

    return mean, dot_product
