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

from .data_analysis import compute_sigma_inverse
from .data_analysis import compute_intersection_kernels


class Pyrotor():
    """
    Optimize your trajectory with Pyrotor.
    """

    def __init__(self,
                 cost_function,
                 reference_trajectories,
                 constraint,
                 end_points,
                 basis,
                 basis_dimension,
                 iteration_setting):
        """
        Create a new Pyrotor optimization

        Inputs:
        """
        pass

    def initialize_ref_coefficients(self):
        self.ref_coefficients = compute_ref_coefficients(self.ref_trajectories,
                                                         self.basis,
                                                         self.var_dim)

    def compute_trajectory(self):
        """
        Compute a trajectory in accordance with aeronautical standards
        """
        self.omega = compute_vector_omega(self.ref_TFC)
        self.ref_coefficients = compute_ref_coefficients(self.ref_trajectories,
                                                         self.longest_ref_climb_duration,
                                                         self.I,
                                                         self.basis,
                                                         self.var_dim)
        self.compute_optimal_trajectory()
        self.is_valid = is_in_constraints(self.y, self.u['VARIO'].values, self.mass, self.protection, self.vmo_mmo)

    def compute_optimal_trajectory(self):
        """
        Compute the optimized trajectory
        """
        # Compute matrices involved on the final cost function
        W, Q = compute_objective_matrices(self.basis,
                                          self.basis_dimension,
                                          self.model_path)
        # Compute the pseudo-inverse of variance-covariance matrix
        sigma_inverse = compute_sigma_inverse(self.ref_coefficients)
        # Compute intersection between ker phi.T*phi and ker sigma
        v_kernel = compute_intersection_kernels()
        # Init endpoints constraints
        get_linear_constraints()
        add_linear_constraints(v_kernel, self.ref_coefficients)
        # ou multiplier vector_omega par kappa
        # Compute the weighted coefficients
        c_weight = compute_weighted_coef(self.ref_coefficients,
                                         self.weights,
                                         self.basis_dimension)
        c_opt = compute_optimized_coefficients(Q,
                                               W,
                                               phi,
                                               lin_const,
                                               sigma_inverse,
                                               c_weight)
        # Construction optimized trajectory from coefficients
        self.y_opt = coef_to_traj(c_opt, self.longest_ref_climb_duration, self.basis, self.basis_dimension)
        self.optimized_cost = compute_cost(self.y_opt)
