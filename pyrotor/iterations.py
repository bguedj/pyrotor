# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Describe the iterative process performed while optimizing trajectories.
"""

import numpy as np

from .cost_functions import compute_f
from .cost_functions import compute_g
from .log import log


def get_kappa_boundaries(reference_coefficients, matrix_q, vector_w,
                         sigma_inverse, c_weight, opti_factor):
    """
    Give the possible minumum and maximum supposed value of kappa.

    Inputs:
        - reference_coefficients: ndarray
            Coefficients of reference
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
            Optimisation factor: How far you want to optimize

    Outputs:
        - kappa_min: float
            Supposed possible minimum value of kappa
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

    kappa_min = compute_kappa_min(kappa_mean)
    kappa_max = compute_kappa_max(kappa_mean, opti_factor)

    return kappa_min, kappa_max


def compute_kappa_min(kappa_mean):
    """
    Compute the supposed possible minimum value of kappa

    Inputs:
        - kappa_mean: float
            Mean kappa

    Output:
        - kappa_min: float
            Supposed possible minimum value of kappa
    """
    return kappa_mean * 0


def compute_kappa_max(kappa_mean, opti_factor):
    """
    Compute the supposed possible maximum value of kappa

        Inputs:
            - kappa_mean: float
                Mean kappa
            - opti_factor: float
                Optimisation factor: How far you want to optimize

        Output:
            - kappa_max: float
                Supposed possible maximum value of kappa
    """
    return opti_factor * kappa_mean


def compute_kappa_mean(evaluations_f, evaluations_g):
    """
    Compute the mean kappa

    Inputs:
        - evaluations_f: list
            Evaluations of several reference coefficients over f
        - evaluations_g: list
            Evaluations of several reference coefficients over g

    Output:
        kappa_mean: float
            Mean kappa
    """
    evaluations_f = np.array(evaluations_f)
    evaluations_g = np.array(evaluations_g)
    kappa = np.abs(evaluations_f/evaluations_g)
    return np.mean(kappa)


def iterate_through_kappas(trajectory, kappa_min, kappa_max, verbose):
    """
    Iterate through the different kappas in order to find the optimum
    trajectory that follows our constraints.

    Inputs:
        - trajectory: Pyrotor instance
            Trajectory to optimize.
        - kappa_min: float
            Supposed minimum possible kappa
        - kappa_max: float
            Supposed maximum possible kappa
    """
    trajectory.kappas = np.linspace(kappa_min, kappa_max, 100000)
    trajectory.i_binary_search = 0
    binary_search_best_trajectory(trajectory,
                                  len(trajectory.kappas)-1,
                                  len(trajectory.kappas)-1,
                                  verbose)
    if not trajectory.is_valid:
        raise ValueError("Trajectories of reference too close to the constraints:\nAborted")


def binary_search_best_trajectory(trajectory, i, step, verbose):
    """
    Perfor a binary search amoung all the kappas to find the best trajectory

    Inputs:
        - i: int
            index of the kappa to use
        - step: int
            size of the current split
    """
    trajectory.i_kappa = i
    trajectory.i_binary_search += 1

    if i < 0:
        raise ValueError("Trajectories of reference too close to the constraints:\nAborted")

    trajectory.kappa = trajectory.kappas[i]
    trajectory.compute_one_iteration()
    log("Trajectory cost: {}".format(trajectory.cost), verbose)

    step = step//2
    if not trajectory.is_valid:
        log('The trajectory found doesn\'t satisfy the constraints. Continue', verbose)
        if step == 0:
            step = 1
        binary_search_best_trajectory(trajectory, i-step, step, verbose)
    else:
        if len(trajectory.kappas)-1 != i and step != 0:
            log('The trajectory found satisfies the constraints. Continue', verbose)
            binary_search_best_trajectory(trajectory, i+step, step, verbose)
        else:
            log('The trajectory found satisfies the constraints.', verbose)
            log('Optimal solution found. Finishing...', verbose)
