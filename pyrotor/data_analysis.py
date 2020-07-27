# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
File to analyze the reference trajectories
"""

import numpy as np

from sklearn.covariance import EmpiricalCovariance
from sklearn.covariance import GraphicalLasso


def nb_samples_is_sufficient(X):
    if np.shape(X)[0] > 2 * np.shape(X)[1]:
        return True
    else:
        return False


def compute_covariance(X):
    """
    Estimate covariance and precision matrices from data X - Depending on
    samples number, use either EmpiricalCovariance or GraphicalLasso methods
    from scikit-learn

    Input:
        X: ndarray
            Data
    
    Outputs:
        covariance: ndarray
            Estimated covariance matrix
        precision: ndarray
            Estimated precision matrix (i.e. pseudo-inverse of covariance)
    """
    if nb_samples_is_sufficient(X):
        cov = EmpiricalCovariance().fit(X)
    else:
        cov = GraphicalLasso(mode='lars').fit(X)
    covariance = cov.covariance_
    precision = cov.precision_

    return covariance, precision


def compute_intersection_kernels(A, B):
    """
    """
