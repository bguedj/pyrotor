# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
File containing optimization solvers.
"""

import numpy as np

from cvxopt import solvers
from cvxopt import matrix

from scipy.optimize import minimize, approx_fprime
from scipy.integrate import trapz

from .projection import coef_to_trajectory


def compute_optimized_coefficients(format_model, phi, lin_const, sigma_inverse,
                                   c_weight, kappa, quadratic_model,
                                   extra_info,
                                   use_quadratic_programming=True):
    """
    Depending on quadratic_model and use_quadratic_programming, decide which
    solver to use.

    Inputs:
        - format_model: list of arrays or sklearn model
            Model of the cost; if list, the first element of the list is the
            integrated linear part W and the second one is the integrated
            quadratic part q
        - phi: ndarray
            Matrix representing the initial and final linear conditions
        - lin_const: scipy.optimize.LinearConstraint
            Object containing the initial and final conditions
        - sigma_inverse: ndarray
            Pseudoinverse of the covariance matrix of the reference
            coefficients
        - c_weight: ndarray
            Coefficients of a weighted trajectory
        - kappa: float
            Optimisation hyperparameter
        - quadratic_model: bool
            Indicate if the model is quadratic
        - extra_info: dict
            Contains independent_variable, basis, basis_features and
            basis_dimension dictionaries
        - use_quadratic_programming: Boolean
            Use or not quadratic programming solver
    Return:
        - c_opt: list of arrays
            Coefficients of the optimized trajectory
    """
    if quadratic_model and use_quadratic_programming:
        W, Q = format_model[0], format_model[1]
        # Use CVXOPT library
        c_opt = minimize_cvx(Q, W, phi, lin_const, sigma_inverse, c_weight,
                             kappa)
    else:
        # Use scipy.optimize.minimize(method='trust-constr')
        c_opt = minimize_trust_constr(format_model, lin_const, sigma_inverse,
                                      c_weight, kappa, quadratic_model,
                                      extra_info)
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


def minimize_trust_constr(model, lin_const, sigma_inverse, c_weight,
                          kappa, quadratic_model, extra_info):
    """
    Minimize a function using scipy.optimize.

    Inputs:
        - model: list of arrays or sklearn model
            Model of the cost; if list, the first element of the list is the
            integrated linear part W and the second one is the integrated
            quadratic part q
        - lin_const: scipy.optimize.LinearConstraint
            Object containing the initial and final conditions
        - sigma_inverse: ndarray
            Pseudoinverse of the covariance matrix of the reference
            coefficients
        - c_weight: ndarray
            Coefficients of a weighted trajectory
        - kappa: float
            Optimisation hyperparameter
        - quadratic_model: bool
            Indicate if the model is quadratic
        - extra_info: dict
            Contains independent_variable, basis, basis_features and
            basis_dimension dictionaries
    Output:
        c_optimized: ndarray
            Coefficients of the optimized trajectory
    """
    # If quadratic model, define explicit cost function to speed the
    # algorithm up
    if quadratic_model:
        W, Q = model[0], model[1]
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
    # Else compute cost by reconstructing first the trajectory and by
    # computing then the cost via the model
    else:
        # Get information to reconstruct the trajectory
        independent_variable = extra_info['independent_variable']
        basis = extra_info['basis']
        basis_features = extra_info['basis_features']
        basis_dimension = extra_info['basis_dimension']
        start = independent_variable['start']
        end = independent_variable['end']
        points_nb = independent_variable['points_nb']

        # Define the cost
        def cost(x, model, sigma_inverse, kappa):
            traj = coef_to_trajectory(x, points_nb, basis, basis_features,
                                      basis_dimension)
            # Define the non-regularised cost
            cost_inst = kappa * model.predict(traj)
            cost_tot = trapz(cost_inst, np.linspace(start, end, points_nb))
            # Add regularisation
            cost_tot += np.linalg.multi_dot([x.T, sigma_inverse, x])
            cost_tot -= 2 * np.linalg.multi_dot([c_weight.T,
                                                 sigma_inverse,
                                                 x])
            return cost_tot

        # Compute jacobian using scipy.approx_fprime
        def cost_jac(x, cost, model, sigma_inverse, kappa):
            def f(x): return cost(x, model, sigma_inverse, kappa)
            return approx_fprime(x, f, epsilon=1e-6)
        # Use scipy.optimize.minimize(method='trust-constr')
        res = minimize(lambda x: cost(x, model, sigma_inverse, kappa),
                       c_weight,
                       method='trust-constr',
                       jac=lambda x: cost_jac(x,
                                              cost,
                                              model,
                                              sigma_inverse,
                                              kappa),
                       constraints=lin_const,
                       options={'maxiter': 300, 'verbose': 1})
    c_optimized = np.array(res.x).ravel()

    return c_optimized
