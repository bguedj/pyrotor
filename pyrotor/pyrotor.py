# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Class for trajectory optimization
"""

import numpy as np
import pandas as pd

from .projection import trajectories_to_coefs
from .projection import coef_to_trajectory
from .projection import compute_weighted_coef

from .initial_and_final_states import get_linear_endpoints

from .constraints import is_in_constraints

from .objective_matrices import compute_objective_matrices

from .cost_functions import compute_cost
from .cost_functions import compute_trajectories_cost

from .data_analysis import compute_covariance
from .data_analysis import compute_intersection_kernels
from .data_analysis import compute_weights
from .data_analysis import select_trajectories

from .iterations import get_kappa_boundaries
from .iterations import iterate_through_kappas

from .optimization import compute_optimized_coefficients


class Pyrotor():
    """
    Optimize your trajectory with Pyrotor.
    """

    def __init__(self,
                 quadratic_model,
                 reference_trajectories,
                 endpoints,
                 constraints,
                 basis,
                 basis_dimension,
                 iteration_setting,
                 independent_variable,
                 n_best_trajectory_to_use=10,
                 verbose=True):
        """
        Create a new Pyrotor optimization

        Inputs:
        """
        self.quadratic_model = quadratic_model
        self.reference_trajectories = reference_trajectories
        self.endpoints = endpoints
        self.constraints = constraints
        self.basis = basis
        self.basis_dimension = basis_dimension
        self.iteration_setting = iteration_setting
        self.n_best_trajectory_to_use = n_best_trajectory_to_use
        self.verbose = verbose

        self.initialize_ref_coefficients()
        self.reference_costs = compute_trajectories_cost(self.reference_trajectories,
                                                         self.quadratic_model)
        # Compute matrices involved on the final cost function
        self.W, self.Q = compute_objective_matrices(self.basis,
                                          self.basis_dimension,
                                          self.quadratic_model)
        # Compute the pseudo-inverse of variance-covariance matrix
        _, self.sigma_inverse = compute_covariance(self.ref_coefficients)

        # Compute intersection between ker phi.T*phi and ker sigma
        # self.v_kernel = compute_intersection_kernels()
        # add_linear_constraints(v_kernel, self.ref_coefficients)
        # Init endpoints constraints
        self.linear_constraints, self.phi = get_linear_endpoints(self.basis_dimension,
                                                                 self.endpoints)
        self.reference_trajectories = select_trajectories(self.reference_trajectories,
                                                          self.reference_costs,
                                                          self.n_best_trajectory_to_use)
        self.initialize_ref_coefficients()
        self.reference_costs = compute_trajectories_cost(self.reference_trajectories,
                                                         self.quadratic_model)

        self.weights = compute_weights(self.reference_costs)
        self.c_weight = compute_weighted_coef(self.ref_coefficients,
                                         self.weights,
                                         self.basis_dimension)
        self.kappa_min, self.kappa_max = get_kappa_boundaries(self.ref_coefficients, self.Q, self.W,
                                                              self.sigma_inverse, self.c_weight)
        self.independent_variable = independent_variable

    def initialize_ref_coefficients(self):
        self.ref_coefficients = trajectories_to_coefs(self.reference_trajectories,
                                                      self.basis,
                                                      self.basis_dimension)

    def compute_one_iteration(self):
        """
        Compute a trajectory in accordance with aeronautical standards
        """
        try:
            c_opt = compute_optimized_coefficients(self.Q,
                                                   self.W,
                                                   self.phi,
                                                   self.linear_constraints,
                                                   self.sigma_inverse,
                                                   self.c_weight,
                                                   self.kappa)
            # Construction optimized trajectory from coefficients
            self.trajectory = coef_to_trajectory(c_opt,
                                                 self.independent_variable["points_nb"],
                                                 self.basis,
                                                 self.basis_dimension)
            self.is_valid = is_in_constraints(self.trajectory, self.constraints)
            self.trajectory_cost = compute_cost(self.trajectory,
                                                self.quadratic_model)
        except ValueError:
            self.is_valid = False

    def compute_optimal_trajectory(self):
        """
        Compute the optimized trajectory
        """
        iterate_through_kappas(self, self.kappa_min, self.kappa_max,
                               self.verbose)
        # self.optimized_cost = compute_cost(self.trajectory)

    def compute_gains(self):
        return self.reference_costs - self.trajectory_cost

    def compute_relative_gains(self):
        gains = (self.reference_costs - self.trajectory_cost) / self.reference_costs
        return gains
