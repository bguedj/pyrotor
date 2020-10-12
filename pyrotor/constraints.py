# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Constraints verification.
"""


def is_in_constraints(trajectory, constraints, costs_by_time):
    """
    Check wether or not the trajectory is complying with each
    constraint.

    Inputs:
         - trajectory: pandas DataFrame
              Trajectory to be checked
         - constraints: list
              Constraints the trajectory must complain with.
              Each constraint is model as a function; if the function
              is negative when applied on the right variable, then the
              constraint is considered as not satisfied.
              ex: [f1, f2] and if f1(trajectory) < 0 then the
              constraint is not satisfied
         - costs_by_time: ndarray
              Optimized trajectory cost (useful when the constraints
              depend on the trajectory cost)
    Output:
         is_in: bool
              Wether or not constraints are satisfied
    """
    # Create column for cost
    trajectory_and_cost = trajectory
    trajectory_and_cost["cost"] = costs_by_time
    for constraint in constraints:
        evaluation = constraint(trajectory_and_cost)
        if (evaluation <= 0).any():
            return False
    return True
