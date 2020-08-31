# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Compute matrices appearing in the cost function
"""

import pickle

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from .basis import compute_legendre_features


def model_to_matrix(model, basis_dimension):
    """
    From a quadratic model f(x), compute q and w such that
    f(x) = <x, qx> + <w, x> + r
    The model has to be in pickle format and previously trained using the
    pipeline [PolynomialFeatures(), StandardScaler(), LinearRegressor()] from
    scikit-learn

    Inputs:
        - model: sklearn.pipeline.Pipeline
            Quadratic model
        - basis_dimension: dict
            Give the number of basis functions for each variable

    Outputs:
        - w: ndarray
            Vector of the linear term (without intercept)
        - q: ndarray
            Matrix of the quadratic term
    """
    # Compute number of variables
    n_var = len(basis_dimension)

    # FIXME: do not use "named_steps" for compatibility concerns
    # Get coefficients of the model (from make_pipeline of sklearn)
    coef = np.array(model.named_steps['lin_regr'].coef_)
    # Remove normalization from StandardScaler()
    std_ = np.sqrt(model.named_steps['scale'].var_)
    coef /= std_
    # Define w
    w = coef[1:n_var+1]
    # Define q starting by the upper part and deduce then the lower one
    coef = np.delete(coef, range(n_var+1))
    # Divide coef by two because a x^2 + b xy + c y^2 is associated with
    # [[a, b/2],[b/2, c]]
    coef /= 2
    q = np.zeros([n_var, n_var])
    for i in range(n_var):
        q[i, i:] += coef[:n_var - i]
        # Mutliply the diagonal by 2
        q[i, i] *= 2
        coef = np.delete(coef, range(n_var - i))
    # Deduce the lower part
    q += np.transpose(np.triu(q, 1))
    return w, q


def extend_matrix(w, q, mean, dot_product, basis_dimension):
    """
    Extend respectively matrix and vector from a quadratic model to new matrix
    and vector with constant blocks - Their size is given by basis_dimension

    Inputs:
        - w: ndarray
            Vector of the linear term of the quadratic model
        - q: ndarray
            Matrix of the quadratic term of the quadratic model
        - mean: ndarray
            Array containing the means over an interval of each element of a
            functional basis
        - dot_product: numpy array [d, d]
            Matrix containing the dot products between each element of a
            functional basis
        - basis_dimension: dict
            Give the number of basis functions for each variable

    Outputs:
        - W: ndarray
            Extended vector from the linear term (without intercept)
        - Q: ndarray
            Extended matrix from the quadratic term
    """
    # Compute the dimension of the problem
    d = np.sum([basis_dimension[elt] for elt in basis_dimension])
    # Extend w to a constant block vector
    W_ = np.zeros(d)
    k = 0
    for i1, var in enumerate(basis_dimension):
        for i2 in range(basis_dimension[var]):
            W_[i2 + k] += w[i1]
        k += basis_dimension[var]
    # Multiply W_ by mean to obtain W
    W = np.multiply(W_, mean)
    # Extend q to a constant block matrix
    Q_ = np.zeros([d, d])
    k, l = 0, 0
    for i1, var1 in enumerate(basis_dimension):
        for j1, var2 in enumerate(basis_dimension):
            for i2 in range(basis_dimension[var1]):
                for j2 in range(basis_dimension[var2]):
                    Q_[i2 + k, j2 + l] += q[i1, j1]
            l += basis_dimension[var2]
        l = 0
        k += basis_dimension[var1]
    # Multiply Q_ by dot_product to obtain Q
    Q = np.multiply(Q_, dot_product)

    return W, Q


def compute_objective_matrices(basis, basis_dimension, quad_model):
    """
    Compute the matrices and vectors from a quadratic model and involved in
    the final cost function

    Inputs:
        - basis: string
            Name of the functional basis
        - basis_dimension: dict
            Give the number of basis functions for each variable
        - quad_model: str or list
            if str then it is the path to the folder containing the pickle
            model; else the first element of the list is w and the second one
            is q

    Outputs:
        - W: ndarray
            Vector involved in the linear part of the cost function
        - Q: ndarray
            Matrix involved in the quadratic part of the cost function
    """
    # Compute means and dot products depending on the basis
    if basis == 'legendre':
        mean, dot_product = compute_legendre_features(basis_dimension)
    # If pickle model, compute w, q using model_to_matrix()
    if isinstance(quad_model, Pipeline):
        # Compute w, q associated with the quadratic model
        w, q = model_to_matrix(quad_model, basis_dimension)
    # Else extract w, q from quad_model
    else:
        w, q = quad_model[1], quad_model[2]
    # Compute W, Q appearing in the final cost function
    W, Q = extend_matrix(w, q, mean, dot_product, basis_dimension)
    return W, Q
