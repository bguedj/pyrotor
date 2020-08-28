import unittest
import numpy as np
import pandas as pd

from pyrotor.constraints import is_in_constraints


def test_is_in_constraints():
    trajectory = pd.DataFrame({"A": [1.1, 2, 3, 4],
                               "B": [9, 6, 1, 0.25]})
    cost_by_time = np.array([2.7, 3.1, 3.4, 5.9])
    # A has to be greater than 1
    def f1(trajectory):
        return (trajectory["A"].values > 1).all()
    # A has to be smaller than 5
    def f2(trajectory):
        return -trajectory["A"].values + 5
    # B has to be smaller than x²
    def f3(trajectory):
        return np.arange(len(trajectory)) ** 2 - trajectory["B"].values

    constraints = [f1, f2]

    expected = True

    assert is_in_constraints(trajectory, constraints, cost_by_time) == expected

    constraints = [f1, f2, f3]
    expected = False

    assert is_in_constraints(trajectory, constraints, cost_by_time) == expected
