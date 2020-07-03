# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Class for trajectory optimization
"""

import numpy as np
import pandas as pd

from .projection import trajectories_to_coefs, coef_to_trajectory
from .initial_and_final_states import get_linear_endpoints
from .constraints import is_in_constraints


class Pyrotor():
    """
    Opti class contains all the routines to optimize a flight
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

    def initialize_ref_climbs(self):
        self.ref_coefficients = compute_ref_coefficients(self.ref_trajectories,
                                                         self.basis,
                                                         self.var_dim)

    def compute_trajectory(self):
        """
        Compute a trajectory in accordance with aeronautical standards
        """
        self.vector_omega = compute_vector_omega(self.ref_TFC)
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
        # Init objects for optimization
        W, Q, D2 = compute_matrices_cost_function(self.longest_ref_climb_duration,
                                                        self.var_dim,
                                                        self.model_path,
                                                        self.basis)
        lambd = weight_components(self.var_dim, self.weight_dict)
        ref_coefficients = compute_weighted_coefficient(self.var_dim, self.ref_coefficients, self.vector_omega)
        # Init endpoints constraints
        get_linear_constraints()
        # ou multiplier vector_omega par kappa
        c_opt = compute_optimized_coefficients(Q, W, Phi, linear_constraints, lambd, D2,
                                           ref_coefficients, self.vector_omega,
                                           self.quadratic_programming)
        # Construction optimized flight from coefficients
        self.y_opt = coef_to_traj(c_opt, self.longest_ref_climb_duration, self.basis, self.basis_dimension)
        self.optimized_cost = compute_cost(self.y_opt)
