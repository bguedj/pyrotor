# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Describe the iterative process performed to find and optimized
trajectory that satisfies the constraints.
"""

import numpy as np

from .cost_functions import compute_f
from .cost_functions import compute_g
from .log import log


def get_kappa_boundaries(reference_coefficients, matrix_q, vector_w,
                         sigma_inverse, c_weight, opti_factor):
    """
    Give the possible minumum and maximum supposed values of kappa.

    Here the cost function is of the form f + kappa*g so kappa is
    chosen in such a way that kappa = opti_factor * mean f / mean g
    where opti_factor is chosen by the user.

    The larger opti_factor is, the more important g is when optimising.

    Inputs:
        - reference_coefficients: ndarray
            Coefficients of reference trajectories
        - vector_q: ndarray
            Matrix of the quadratic term
        - vector_w: ndarray
            Vector of the linear term (without intercept)
        - sigma_inverse: ndarray
            Pseudoinverse of the covariance matrix of the reference
            coefficients
        - c_weight: ndarray
            Coefficients of a weighted trajectory
        - opti_factor: float
            Optimisation factor
    Outputs:
        - kappa_min: float
            Supposed possible minimum value of kappa; here set to 0
        - kappa_max: float
            Supposed possible maximum value of kappa
    """
    evaluations_f = []
    evaluations_g = []
    for reference_coefficient in reference_coefficients:
        evaluations_f.append(compute_f(reference_coefficient, sigma_inverse,
                                       c_weight))
        evaluations_g.append(compute_g(reference_coefficient, matrix_q,
                                       vector_w))
    kappa_mean = compute_kappa_mean(evaluations_f, evaluations_g)

    kappa_min = 0
    kappa_max = compute_kappa_max(kappa_mean, opti_factor)

    return kappa_min, kappa_max


def compute_kappa_max(kappa_mean, opti_factor):
    """
    Compute the supposed possible maximum value of kappa.

    Inputs:
        - kappa_mean: float
            Mean of kappa
        - opti_factor: float
            Optimisation factor
    Output:
        - kappa_max: float
            Supposed possible maximum value of kappa
    """
    return opti_factor * kappa_mean


def compute_kappa_mean(evaluations_f, evaluations_g):
    """
    Compute the mean of kappa.

    Inputs:
        - evaluations_f: list
            Evaluations of f over reference coefficients
        - evaluations_g: list
            Evaluations of g over reference coefficients
    Output:
        kappa_mean: float
            Mean of kappa
    """
    evaluations_f = np.array(evaluations_f)
    evaluations_g = np.array(evaluations_g)
    kappa = np.abs(evaluations_f / evaluations_g)
    
    return np.mean(kappa)


def iterate_through_kappas(trajectory, kappa_min, kappa_max, verbose):
    """
    Iterate through the different kappas in order to find the
    optimised trajectory that verifies constraints.

    Inputs:
        - trajectory: Pyrotor instance
            Trajectory to optimize with associated attributes; the
            object will be directly modified by reference
        - kappa_min: float
            Supposed minimum possible kappa
        - kappa_max: float
            Supposed maximum possible kappa
        - verbose: boolean
            Display or not the verbose
    """
    # Define the linear grid for kappas
    trajectory.kappas = np.linspace(kappa_min, kappa_max, 100000)
    # Start at 0
    trajectory.i_binary_search = 0
    binary_search_best_trajectory(trajectory,
                                  len(trajectory.kappas)-1,
                                  len(trajectory.kappas)-1,
                                  verbose)
    if not trajectory.is_valid:
        raise ValueError("Trajectories of reference too close to the \
                         constraints:\nAborted")


def binary_search_best_trajectory(trajectory, i, step, verbose):
    """
    Perform a binary search amoung all the kappas in order to find the
    optimised trajectory that verifies constraints.

    Inputs:
        - trajectory: Pyrotor instance
            Trajectory to optimize with associated attributes; the
            object will be directly modified by reference
        - i: int
            Index of the kappa to use
        - step: int
            Size of the current split
        - verbose: boolean
            Display pr not the verbose
    """
    # TODO: Add comments ?
    trajectory.i_kappa = i
    trajectory.i_binary_search += 1

    if i < 0:
        raise ValueError("Trajectories of reference too close to the \
                         constraints:\nAborted")

    trajectory.kappa = trajectory.kappas[i]
    trajectory.compute_one_iteration()
    log("Trajectory cost: {}".format(trajectory.cost), verbose)

    step = step//2
    if not trajectory.is_valid:
        log('The trajectory found doesn\'t satisfy the constraints. Continue',
            verbose)
        if step == 0:
            step = 1
        binary_search_best_trajectory(trajectory, i-step, step, verbose)
    else:
        if (len(trajectory.kappas)-1 != i) and (step != 0):
            log('The trajectory found satisfies the constraints. Continue',
                verbose)
            binary_search_best_trajectory(trajectory, i+step, step, verbose)
        else:
            log('The trajectory found satisfies the constraints.', verbose)
            log('Optimal solution found. Finishing...', verbose)
