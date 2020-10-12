# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
File containing optimization solvers.
"""

import numpy as np

from cvxopt import solvers
from cvxopt import matrix

from scipy.optimize import minimize


def compute_optimized_coefficients(Q, W, phi, lin_const, sigma_inverse,
                                   c_weight, kappa,
                                   use_quadratic_programming=True):
    """
    Depending on use_quadratic_programming, decide which solver to use.

    Inputs:
        Q: ndarray
            Matrix from the quadratic term of the quadratic model
        W: ndarray
            Vector from the linear term of the quadratic model
        phi: ndarray
            Matrix representing the initial and final linear conditions
        lin_const: scipy.optimize.LinearConstraint
            Object containing the initial and final conditions
        sigma_inverse: ndarray
            Pseudoinverse of the covariance matrix of the reference
            coefficients
        c_weight: ndarray
            Coefficients of a weighted trajectory
        kappa: float
            Optimisation hyperparameter
        use_quadratic_programming: Boolean
            Use or not quadratic programming solver
    Return:
        c_opt: list of arrays
            Coefficients of the optimized trajectory
    """
    if use_quadratic_programming:
        # Use CVXOPT library
        c_opt = minimize_cvx(Q, W, phi, lin_const, sigma_inverse, c_weight,
                             kappa)
    else:
        # Use scipy.optimize.minimize(method='trust-constr')
        c_opt = minimize_trust_constr(Q, W, phi, lin_const, sigma_inverse,
                                      c_weight, kappa)
    c_opt = list(c_opt)

    return c_opt


def minimize_cvx(Q, W, phi, lin_const, sigma_inverse, c_weight, kappa):
    """
    Minimize a quadratic function using CVXOPT library.

    The associated matrix must be positive semidefinite.

    Inputs:
        Q: ndarray
            Matrix from the quadratic term of the quadratic model
        W: ndarray
            Vector from the linear term of the quadratic model
        phi: ndarray
            Matrix representing the initial and final linear conditions
        lin_const: scipy.optimize.LinearConstraint
            Object containing the initial and final conditions
        sigma_inverse: ndarray
            Pseudoinverse of the covariance matrix of the reference
            coefficients
        c_weight: ndarray
            Coefficients of a weighted trajectory
        kappa: float
            Optimisation hyperparameter
    Output:
        c_optimized: ndarray
            Coefficients of the optimized trajectory
    """
    # Define quadratic part
    P = matrix(2 * (kappa * Q + sigma_inverse))
    # Define linear part
    q = matrix((kappa * W - 2 * np.dot(sigma_inverse, c_weight)).T)
    # Define upper and lower endpoints matrix
    # NB: CVXOPT does not deal with two-sided conditions so modify
    # matrix and lower-upper_bounds
    phi = np.concatenate([phi, -phi], axis=0)
    G = matrix(phi, tc='d')
    # Define linear conditions
    lb = -np.array(lin_const.lb)
    h = np.concatenate((lin_const.ub, lb), axis=0)
    h = matrix(h, tc='d')
    # Do not display CVXOPT verbose
    solvers.options['show_progress'] = False
    # Solve using qp method
    optimize = solvers.qp(P, q, G, h)
    c_optimized = np.array(optimize['x']).ravel()

    return c_optimized


def minimize_trust_constr(Q, W, phi, lin_const, sigma_inverse, c_weight,
                          kappa):
    """
    Minimize a quadratic function using scipy.optimize.

    The associated matrix is not required to be positive semidefinite.

    Inputs:
        Q: ndarray
            Matrix from the quadratic term of the quadratic model
        W: ndarray
            Vector from the linear term of the quadratic model
        phi: ndarray
            Matrix representing the initial and final linear conditions
        lin_const: scipy.optimize.LinearConstraint
            Object containing the initial and final conditions
        sigma_inverse: ndarray
            Pseudoinverse of the covariance matrix of the reference
            coefficients
        c_weight: ndarray
            Coefficients of a weighted trajectory
        kappa: float
            Optimisation hyperparameter

    Output:
        c_optimized: ndarray
            Coefficients of the optimized trajectory
    """
    # Define quadratic part
    P = kappa * Q + sigma_inverse
    # Define linear part
    q = kappa * W - 2 * np.dot(sigma_inverse, c_weight)

    # Compute cost together with Jacobian and Hessian
    def cost(x):
        return np.linalg.multi_dot([x.T, P, x]) + np.dot(q, x)

    def cost_jac(x):
        return 2 * np.dot(P, x) + q

    def cost_hess(x, p):
        return 2 * np.dot(P, p)

    # Use scipy.optimize.minimize(method='trust-constr')
    res = minimize(cost, c_weight,
                   method='trust-constr',
                   jac=cost_jac,
                   hessp=cost_hess,
                   constraints=lin_const,
                   options={'verbose': 1})
    c_optimized = np.array(res.x).ravel()

    return c_optimized
