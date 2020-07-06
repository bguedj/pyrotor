# !/usr/bin/env python
# -*- coding:utf-8 -*-


"""
Constraints verification
"""


def is_in_constraints(trajectory, constraints):
    """
    Check wether or not the trajectory is complying to each constraints.

    Inputs:
        - trajectory: pandas DataFrame
             Trajectory to be checked
        - constraints: dict of list
             Constraints the trajectory must complain with.
             Each constraint is model as a function. If the funciton is
             negative when applied on the right variable, then the constraint
             is considered as not satisfied.
             ex: {"var 1": [f1, f2], ...} where "var 1" is one of the columns of
             "trajectory". and if f1(trajectory["var 1"]) < 0 then the
             constraint is not respected.

    Output:
        is_in: bool
             Wether or not it is in constraints
    """
    for variable_name in constraints:
        is_in = variable_is_in_constraints(trajectory[variable_name],
                                           constraints[variable_name])
        if not is_in:
            return False
    return True


def variable_is_in_constraints(variable, constraints):
    """
    Check wether or not the variabme is complying to each constraints.

    Inputs:
        - variable: pandas Series or ndarray
             Variable to be checked
        - constraints: list
             Constraints the variable must complain with.
             Each constraint is model as a function. If the funciton is
             negative when applied on the right variable, then the constraint
             is considered as not satisfied.
             ex: [f1, f2]
             If f1(variable) < 0 then the constraint is not respected.

    Output:
        is_in: bool
             Wether or not it is in constraints
    """
    for constraint in constraints:
        y = constraint(variable.values)
        if (y < 0).any():
            return False
    return True
