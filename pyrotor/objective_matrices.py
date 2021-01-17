# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Compute matrices involved in the cost function
"""

import pickle

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from .basis import compute_basis_features


def model_to_matrix(model, basis_dimension, derivative):
    """
    From a quadratic model f(x), compute q and w such that
    f(x) = <x, qx> + <w, x> + r .
    The model has to be in pickle format and previously trained using
    the pipeline
    [PolynomialFeatures(), StandardScaler(), LinearRegressor()]
    from scikit-learn.

    Inputs:
        - model: sklearn.pipeline.Pipeline
            Quadratic model
        - basis_dimension: dict
            Give the number of basis functions for each state
        - derivative: boolean
            Take into account or not derivatives of states
    Outputs:
        - w: ndarray
            Vector of the linear term (without intercept)
        - q: ndarray
            Matrix of the quadratic term
    """
    # If derivative, add dimensions of derivative into basis_dimension
    if derivative:
        basis_dim_deriv = {}
        for state in basis_dimension.keys():
            deriv_state = state + '_deriv'
            basis_dim_deriv[deriv_state] = basis_dimension[state]
        basis_dimension = {**basis_dimension, **basis_dim_deriv}
    n_var = len(basis_dimension)
    # FIXME: Do not use "named_steps" for compatibility concerns
    # Get coefficients of the model (from make_pipeline of sklearn)
    coef = np.array(model.named_steps['lin_regr'].coef_)
    # Remove normalization from StandardScaler()
    std_ = np.sqrt(model.named_steps['scale'].var_)
    coef[1:] /= std_[1:]

    w = coef[1:n_var+1]

    # Define q starting by the upper part and deduce then the lower one
    coef = np.delete(coef, range(n_var+1))
    # Divide coef by two because a x^2 + b xy + c y^2 is associated with
    # [[a, b/2],[b/2, c]]
    coef /= 2
    q = np.zeros([n_var, n_var])
    for i in range(n_var):
        q[i, i:] += coef[:n_var - i]
        q[i, i] *= 2
        coef = np.delete(coef, range(n_var - i))
    # Deduce the lower part
    q += np.transpose(np.triu(q, 1))

    return w, q


def extend_matrix(w, q, basis_dimension, derivative):
    """
    Extend respectively matrix and vector from a quadratic model to
    new matrix and vector with constant blocks for modelling
    optimization problem at the end.

    Sizes are given by basis_dimension.

    Inputs:
        - w: ndarray
            Vector of the linear term of the quadratic model
        - q: ndarray
            Matrix of the quadratic term of the quadratic model
        - basis_dimension: dict
            Give the number of basis functions for each state
        - derivative: boolean
            Take into account or not derivatives of states
    Outputs:
        - W_: ndarray
            Extended vector from the linear term
        - Q_: ndarray
            Extended matrix from the quadratic term
    """
    # If derivative, add dimensions of derivative into basis_dimension
    if derivative:
        basis_dim_deriv = {}
        for state in basis_dimension.keys():
            deriv_state = state + '_deriv'
            basis_dim_deriv[deriv_state] = basis_dimension[state]
        basis_dimension = {**basis_dimension, **basis_dim_deriv}
    d = np.sum([basis_dimension[elt] for elt in basis_dimension])

    # Extend w to a constant block vector
    W_ = np.zeros(d)
    k = 0
    for i1, var in enumerate(basis_dimension):
        for i2 in range(basis_dimension[var]):
            W_[i2 + k] += w[i1]
        k += basis_dimension[var]

    # Extend q to a constant block matrix
    Q_ = np.zeros([d, d])
    k1, k2 = 0, 0
    for i1, var1 in enumerate(basis_dimension):
        for j1, var2 in enumerate(basis_dimension):
            for i2 in range(basis_dimension[var1]):
                for j2 in range(basis_dimension[var2]):
                    Q_[i2 + k1, j2 + k2] += q[i1, j1]
            k2 += basis_dimension[var2]
        k2 = 0
        k1 += basis_dimension[var1]

    return W_, Q_


def compute_objective_matrices(basis, basis_features, basis_dimension,
                               quad_model, derivative):
    """
    Compute the matrices and vectors from a quadratic model and
    involved in the final cost function.

    Inputs:
        - basis: string
            Name of the functional basis
        - basis_features: dict
            Contain information on the basis for each state
        - basis_dimension: dict
            Give the number of basis functions for each state
        - quad_model: str or list
            if str then it is the path to the folder containing the
            pickle model; else the first element of the list is w and
            the second one is q
        - derivative: boolean
            Take into account or not derivatives of states
    Outputs:
        - W: ndarray
            Vector involved in the linear part of the cost function
        - Q: ndarray
            Matrix involved in the quadratic part of the cost function
    """
    # Compute integrals and dot products depending on the basis
    integral, dot_product = compute_basis_features(basis,
                                                   basis_features,
                                                   basis_dimension,
                                                   derivative)

    # If pickle model, compute w, q using model_to_matrix()
    if isinstance(quad_model, Pipeline):
        # Compute w, q associated with the quadratic model
        w, q = model_to_matrix(quad_model, basis_dimension, derivative)
    # Else extract w, q from quad_model
    else:
        w, q = quad_model[1], quad_model[2]

    # Compute W, Q appearing in the final cost function
    W_, Q_ = extend_matrix(w, q, basis_dimension, derivative)
    W = np.multiply(W_, integral)
    Q = np.multiply(Q_, dot_product)
    # If derivative, one has to sum up blocks of W and Q to take into
    # account derivatives of states in the cost function
    if derivative:
        d = np.sum([basis_dimension[elt] for elt in basis_dimension])
        W1 = W[:d]
        W2 = W[d:]
        Q11 = Q[:d, :d]
        Q12 = Q[:d, d:]
        Q21 = Q[d:, :d]
        Q22 = Q[d:, d:]
        W = W1 + W2
        Q = Q11 + Q12 + Q21 + Q22

    return W, Q
