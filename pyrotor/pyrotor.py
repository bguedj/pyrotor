# !/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd

from .projection import trajectories_to_coefs
from .projection import coef_to_trajectory
from .projection import compute_weighted_coef

from .linear_conditions import get_linear_conditions

from .constraints import is_in_constraints

from .objective_matrices import compute_objective_matrices

from .cost_functions import compute_cost
from .cost_functions import compute_cost_by_time
from .cost_functions import compute_trajectories_cost

from .data_analysis import compute_covariance
from .data_analysis import compute_weights
from .data_analysis import select_trajectories
from .data_analysis import add_derivatives

from .iterations import get_kappa_boundaries
from .iterations import iterate_through_kappas

from .optimization import compute_optimized_coefficients


class Pyrotor():
    """
    Main interface of PyRotor, an optimization package based on data.
    """

    def __init__(self,
                 quadratic_model,
                 reference_trajectories,
                 endpoints,
                 constraints,
                 basis,
                 basis_features,
                 independent_variable,
                 n_best_trajectory_to_use=10,
                 opti_factor=2,
                 sigma=None,
                 derivative=False,
                 use_quadratic_programming=True,
                 n_jobs=None,
                 verbose=True):
        """
        Create a new PyPotor class.

        Inputs:
            - quadratic_model: str or list
                if str then it is the path to the folder containing the
                pickle model; else the first element of the list is w,
                the second one is q and the third is the constant
                Ex: (np.array([[1, 0], [2, 3]]), np.array([2, 1]), 8)
            - reference_trajectories: list of DataFrame
                List of reference trajectories
            - endpoints: dict
                Initial and final states that the optimized trajectory
                must satisfy
                ex: {'Var 1': {'start': 109, 'end': 98, 'delta': 10},
                     ...}
            - constraints: list
                Constraints the trajectory must complain with.
                Each constraint is model as a function; if the function
                is negative when applied on the right variable, then
                the constraint is considered as not satisfied.
                ex: [f1, f2] and if f1(trajectory) < 0 then the
                constraint is not satisfied
            - basis: string
                Name of the functional basis
            - basis_features: dict
                Contain information on the basis for each state
            - independent_variable: dict
                Describe the time-interval on which are defined the
                trajectories
                ex: {'start': 0, 'end': 1, 'frequency':.1}
            - n_best_trajectory_to_use: int, default=10
                Number of trajectories to keep
            - opti_factor: float, default=2
                Optimisation factor
            - sigma: ndarray, default=None
                Matrix interpreted as an estimated covariance matrix
            - derivative: boolean, default=False
                Compute the derivative or not
            - use_quadratic_programming: boolean, default=True
                Use or not quadratic programming solver
            - n_jobs: int, default=None
                Number of process to use - If None, sequential
            - verbose: boolean, default=True
                Display the verbose or not
        """
        self.quadratic_model = quadratic_model
        self.reference_trajectories = reference_trajectories
        self.constraints = constraints
        self.basis = basis
        self.basis_features = basis_features
        self.independent_variable = independent_variable
        self.derivative = derivative
        self.use_quadratic_programming = use_quadratic_programming
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Create basis_dimension dictionary containing dimension of each state
        if basis == 'legendre':
            self.basis_dimension = basis_features
        elif basis == 'bspline':
            nb_knots = len(basis_features['knots'])
            # Number of B-splines = number of internal knots + degree + 1
            self.basis_dimension = {i: basis_features[i] + nb_knots + 1
                                    for i in basis_features if i != 'knots'}

        # Compute reference coefficients
        ref_coefficients = trajectories_to_coefs(
            self.reference_trajectories, self.basis, self.basis_features,
            self.basis_dimension, self.n_jobs)
        # If derivative, compute costs with derivatives
        if self.derivative:
            reference_trajectories_deriv = add_derivatives(
                self.reference_trajectories, self.basis_dimension)
            self.reference_costs = compute_trajectories_cost(
                reference_trajectories_deriv, self.quadratic_model)
        else:
            self.reference_costs = compute_trajectories_cost(
                self.reference_trajectories, self.quadratic_model)
        # Compute matrices involved in the final cost function
        self.W, self.Q = compute_objective_matrices(
                self.basis, self.basis_features, self.basis_dimension,
                self.quadratic_model, self.derivative)
        # Get or compute the variance-covariance and precision matrices
        if sigma is not None:
            self.sigma = sigma
            self.sigma_inverse = np.linalg.pinv(self.sigma, hermitian=True)
        else:
            self.sigma, self.sigma_inverse = compute_covariance(
                ref_coefficients)
        # Create endpoints constraints
        self.linear_conditions, self.phi = get_linear_conditions(
            self.basis, self.basis_dimension, endpoints, ref_coefficients,
            self.sigma
            )
        # FIXME: Avoid computing again coefficients and costs ?
        # Select best reference trajectories and compute coefficients and
        # costs
        self.reference_trajectories = select_trajectories(
            self.reference_trajectories, self.reference_costs,
            n_best_trajectory_to_use)
        ref_coefficients = trajectories_to_coefs(
            self.reference_trajectories, self.basis, self.basis_features,
            self.basis_dimension, self.n_jobs)
        # As above, compute derivatives if necessary
        if self.derivative:
            reference_trajectories_deriv = add_derivatives(
                self.reference_trajectories, self.basis_dimension)
            self.reference_costs = compute_trajectories_cost(
                reference_trajectories_deriv, self.quadratic_model)
        else:
            self.reference_costs = compute_trajectories_cost(
                self.reference_trajectories, self.quadratic_model)
        # Compute weights and deduce weighted coefficient and bounds of kappa
        # for optimization
        self.weights = compute_weights(self.reference_costs)
        self.c_weight = compute_weighted_coef(
            ref_coefficients, self.weights, self.basis_dimension)
        self.kappa_min, self.kappa_max = get_kappa_boundaries(
            ref_coefficients, self.Q, self.W, self.sigma_inverse,
            self.c_weight, opti_factor)

    def compute_one_iteration(self):
        """
        Compute an optimised trajectory which must satisfy the
        constraints.
        """
        try:
            c_opt = compute_optimized_coefficients(
                self.Q, self.W, self.phi, self.linear_conditions,
                self.sigma_inverse, self.c_weight, self.kappa,
                self.use_quadratic_programming)
            # Construct optimized trajectory from coefficients
            self.trajectory = coef_to_trajectory(
                c_opt, self.independent_variable["points_nb"], self.basis,
                self.basis_features, self.basis_dimension)
            # Compute costs
            if self.derivative:
                trajectory = add_derivatives(
                    [self.trajectory], self.basis_dimension)
                self.cost_by_time = compute_cost_by_time(
                    trajectory[0], self.quadratic_model)
                self.cost = compute_cost(
                    trajectory[0], self.quadratic_model)
            else:
                self.cost_by_time = compute_cost_by_time(
                    self.trajectory, self.quadratic_model)
                self.cost = compute_cost(
                    self.trajectory, self.quadratic_model)
            self.is_valid = is_in_constraints(
                self.trajectory, self.constraints, self.cost_by_time)
        except ValueError as e:
            print(e)
            self.is_valid = False
            self.cost = np.nan
            self.cost_by_time = np.array([])

    def compute_optimal_trajectory(self):
        """
        Compute the optimized trajectory.
        """
        iterate_through_kappas(self, self.kappa_min, self.kappa_max,
                               self.verbose)
        # self.optimized_cost = compute_cost(self.trajectory)

    def compute_gains(self):
        """
        Compute gains/savings with respect to reference trajectories.
        """
        return self.reference_costs - self.cost

    def compute_relative_gains(self):
        """
        Compute relative gains/savings with respect to reference
        trajectories.
        """
        gains = self.reference_costs - self.cost
        gains /= np.abs(self.reference_costs)

        return gains
