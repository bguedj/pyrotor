# !/usr/bin/env python
# -*- coding:utf-8 -*-


"""
Constraints verification
"""


def is_in_constraints(trajectory, constraints, costs_by_time):
    """
    Check wether or not the trajectory is complying to each constraints.

    Inputs:
        - trajectory: pandas DataFrame
             Trajectory to be checked
        - constraints: list
             Constraints the trajectory must complain with.
             Each constraint is model as a function. If the funciton is
             negative when applied on the right variable, then the constraint
             is considered as not satisfied.
             ex: [f1, f2] and if f1(trajectory) < 0 then the
             constraint is not respected.

    Output:
        is_in: bool
             Wether or not it is in constraints
    """
    trajectory_and_cost = trajectory
    trajectory_and_cost["cost"] = costs_by_time
    for constraint in constraints:
        evaluation = constraint(trajectory_and_cost)
        if (evaluation <= 0).any():
            return False
    return True
