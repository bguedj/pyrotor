import unittest
import numpy as np
import pandas as pd

from pyrotor.constraints import is_in_constraints


def test_is_in_constraints():
    trajectory = pd.DataFrame({"A": [1, 2, 3, 4],
                               "B": [9, 6, 1, 0.25]})
    # A has to be greater than 1
    def f1(trajectory):
        return trajectory["A"].values - 1
    # A has to be smaller than 5
    def f2(trajectory):
        return -trajectory["A"].values + 5
    # B has to be smaller than x²
    def f3(trajectory):
        return np.arange(len(trajectory)) ** 2 - trajectory["B"].values
    
    constraints = [f1, f2]

    expected = True

    assert is_in_constraints(trajectory, constraints) == expected

    constraints = [f1, f2, f3]
    expected = False

    assert is_in_constraints(trajectory, constraints) == expected