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

from .iterations import get_kappa_boundaries
from .iterations import iterate_through_kappas

from .optimization import compute_optimized_coefficients


class Pyrotor():
    """
    Main interface to pyrotor, an optimization package based on data.
    """

    def __init__(self,
                 quadratic_model,
                 reference_trajectories,
                 endpoints,
                 constraints,
                 basis,
                 basis_dimension,
                 independent_variable,
                 n_best_trajectory_to_use=10,
                 opti_factor=2,
                 sigma=None,
                 n_jobs=None,
                 verbose=True):
        """
        Create a new Pyrotor optimization

        Inputs:
            - quadratic_model: tuple or list, it can be a sklearn Pipeline
                Your quadratic model. See example in documentation for more
                details.
            - opti_factor: float
                Optimisation factor: How far you want to optimize
        """
        print("bug?")
        self.quadratic_model = quadratic_model
        self.reference_trajectories = reference_trajectories
        self.constraints = constraints
        self.basis = basis
        self.basis_dimension = basis_dimension
        self.n_jobs = n_jobs
        self.verbose = verbose


        self.ref_coefficients = trajectories_to_coefs(self.reference_trajectories,
                                                      self.basis,
                                                      self.basis_dimension,
                                                      self.n_jobs)

        self.reference_costs = compute_trajectories_cost(self.reference_trajectories,
                                                         self.quadratic_model)
        # Compute matrices involved on the final cost function
        self.W, self.Q = compute_objective_matrices(self.basis,
                                          self.basis_dimension,
                                          self.quadratic_model)
        # Get or compute the variance-covariance and precision matrices
        if sigma is not None:
            self.sigma = sigma
            self.sigma_inverse = np.linalg.pinv(self.sigma, hermitian=True)
        else:
            self.sigma, self.sigma_inverse = compute_covariance(self.ref_coefficients)

        # Compute intersection between ker phi.T*phi and ker sigma
        # self.v_kernel = compute_intersection_kernels()
        # add_linear_conditions(v_kernel, self.ref_coefficients)
        # Init endpoints constraints
        self.linear_conditions, self.phi = get_linear_conditions(self.basis_dimension,
                                                                 endpoints,
                                                                 self.ref_coefficients,
                                                                 self.sigma)
        self.reference_trajectories = select_trajectories(self.reference_trajectories,
                                                          self.reference_costs,
                                                          n_best_trajectory_to_use)
        self.ref_coefficients = trajectories_to_coefs(self.reference_trajectories,
                                                      self.basis,
                                                      self.basis_dimension,
                                                      self.n_jobs)

        self.reference_costs = compute_trajectories_cost(self.reference_trajectories,
                                                         self.quadratic_model)

        self.weights = compute_weights(self.reference_costs)
        self.c_weight = compute_weighted_coef(self.ref_coefficients,
                                         self.weights,
                                         self.basis_dimension)
        self.kappa_min, self.kappa_max = get_kappa_boundaries(self.ref_coefficients, self.Q, self.W,
                                                              self.sigma_inverse, self.c_weight,
                                                              opti_factor)
        self.independent_variable = independent_variable

    def compute_one_iteration(self):
        """
        Compute a trajectory in accordance with aeronautical standards
        """
        try:
            c_opt = compute_optimized_coefficients(self.Q,
                                                   self.W,
                                                   self.phi,
                                                   self.linear_conditions,
                                                   self.sigma_inverse,
                                                   self.c_weight,
                                                   self.kappa)
            # Construction optimized trajectory from coefficients
            self.trajectory = coef_to_trajectory(c_opt,
                                                 self.independent_variable["points_nb"],
                                                 self.basis,
                                                 self.basis_dimension)
            self.cost_by_time = compute_cost_by_time(self.trajectory,
                                                      self.quadratic_model)
            self.cost = compute_cost(self.trajectory,
                                     self.quadratic_model)
            self.is_valid = is_in_constraints(self.trajectory,
                                              self.constraints,
                                              self.cost_by_time)
        except ValueError as e:
            self.is_valid = False
            self.cost = np.nan
            self.cost_by_time = np.array([])

    def compute_optimal_trajectory(self):
        """
        Compute the optimized trajectory
        """
        iterate_through_kappas(self, self.kappa_min, self.kappa_max,
                               self.verbose)
        # self.optimized_cost = compute_cost(self.trajectory)

    def compute_gains(self):
        return self.reference_costs - self.cost

    def compute_relative_gains(self):
        gains = (self.reference_costs - self.cost) / self.reference_costs
        return gains
