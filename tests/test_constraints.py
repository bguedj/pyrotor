import unittest
import numpy as np
import pandas as pd

from pyrotor.constraints import is_in_constraints
from pyrotor.constraints import variable_is_in_constraints


def test_is_in_constraints():
    trajectory = pd.DataFrame({"A": [1, 2, 3, 4],
                               "B": [9, 6, 1, 0.25]})
    # A has to be greater than 1
    def A_f1(x): return x-1
    # A has to be smaller than 5
    def A_f2(x): return -x + 5
    # B has to be smaller than x²
    def B_f1(x): return np.arange(x.shape[0]) ** 2 - x
    constraints = {"A": [A_f1, A_f2]}

    expected = True

    assert is_in_constraints(trajectory, constraints) == expected

    constraints["B"] = [B_f1]
    expected = False

    assert is_in_constraints(trajectory, constraints) == expected


def test_variable_is_in_constraints():
    variable = pd.Series([1, 2, 3, 4], name="A")
    # A has to be greater than 1
    def A_f1(x): return x-1
    # A has to be smaller than 5
    def A_f2(x): return -x + 5

    constraints = [A_f1, A_f2]
    expected = True

    assert variable_is_in_constraints(variable, constraints) == expected

    # A has to be smaller than 5
    def A_f2(x): return -x + 3

    constraints = [A_f1, A_f2]
    expected = False

    assert variable_is_in_constraints(variable, constraints) == expected
