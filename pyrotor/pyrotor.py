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

from .data_analysis import compute_covariance
from .data_analysis import compute_intersection_kernels
from .data_analysis import compute_trajectories_cost
from .data_analysis import compute_weights


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
                 iteration_setting):
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

        self.initialize_ref_coefficients()
        self.reference_costs = compute_trajectories_cost(self.ref_coefficients)

        self.weights = compute_weights(self.reference_costs)
        # Compute matrices involved on the final cost function
        self.W, self.Q = compute_objective_matrices(self.basis,
                                          self.basis_dimension,
                                          self.quadratic_model)
        # Compute the pseudo-inverse of variance-covariance matrix
        self.sigma_inverse = compute_covariance(self.ref_coefficients)
        # Compute intersection between ker phi.T*phi and ker sigma
        self.v_kernel = compute_intersection_kernels()
        # Init endpoints constraints
        self.phi = get_linear_endpoints()
        add_linear_constraints(v_kernel, self.ref_coefficients)
        # ou multiplier vector_omega par kappa
        # Compute the weighted coefficients
        self.c_weight = compute_weighted_coef(self.ref_coefficients,
                                         self.weights,
                                         self.basis_dimension)
        self.kappa_min, self.kappa_max = get_kappa_boundaries(x, Q, W,
                                                              sigma_inverse, c_weight)

    def initialize_ref_coefficients(self):
        self.ref_coefficients = compute_ref_coefficients(self.ref_trajectories,
                                                         self.basis,
                                                         self.basis_dimension)

    def _compute_trajectory(self):
        """
        Compute a trajectory in accordance with aeronautical standards
        """
        c_opt = compute_optimized_coefficients(self.Q,
                                               self.W,
                                               self.phi,
                                               self.lin_const,
                                               self.sigma_inverse,
                                               self.c_weight)
        # Construction optimized trajectory from coefficients
        self.trajectory = coef_to_trajectory(c_opt,
                                             self.longest_ref_climb_duration,
                                             self.basis,
                                             self.basis_dimension)
        self.is_valid = is_in_constraints(self.trajectory, self.constraints)

    def compute_optimal_trajectory(self):
        """
        Compute the optimized trajectory
        """
        iterate_through_kappas(self, self.kappa_min, self.kappa_max)
        self.optimized_cost = compute_cost(self.y_opt)
