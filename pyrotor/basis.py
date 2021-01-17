# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Compute integrals and dot products of functional bases.
"""

import numpy as np
from numpy.polynomial.legendre import Legendre, legmul, legint
from scipy.integrate import quad
from scipy.interpolate import BSpline, make_lsq_spline


def compute_basis_features(basis, basis_features, basis_dimension,
                           derivative):
    """
    Format integrals and dot products of basis elements (and their derivatives)
    for the optimisation problem.

    Inputs:
        - basis: string
            Name of the functional basis
        - basis_features: dict
            Contain information on the basis for each state
        - basis_dimension: dict
            Give the number of basis functions for each state
        - derivative: boolean
            Take into account or not derivatives of states
    Outputs:
        - integral: ndarray
            Array containing the integrals over an interval
        - dot_product: ndarray
            Matrix containing the dot products
    """
    # Compute integrals and dot products only for states
    if basis == 'legendre':
        integral, dot_product = \
            compute_legendre_integral_dot_product(basis_dimension)
    elif basis == 'bspline':
        integral, dot_product = \
            compute_bspline_integral_dot_product(basis_features,
                                                 basis_dimension)
    if derivative:
        # Compute integrals and dot products for derivatives of states
        if basis == 'legendre':
            integral_deriv = \
                compute_legendre_integral_derivatives(basis_dimension)
            dot_product_12, dot_product_22 = \
                compute_legendre_dot_product_derivatives(basis_dimension)
        elif basis == 'bspline':
            integral_deriv = \
                compute_bspline_integral_derivatives(basis_dimension)
            dot_product_12, dot_product_22 = \
                compute_bspline_dot_product_derivatives(basis_features,
                                                        basis_dimension)
        # Concatenate vectors and matrices to format
        integral = np.concatenate((integral, integral_deriv))
        dot_product_1 = np.concatenate((dot_product, dot_product_12), axis=1)
        dot_product_2 = np.concatenate((dot_product_12.T, dot_product_22),
                                       axis=1)
        dot_product = np.concatenate((dot_product_1, dot_product_2), axis=0)

    return integral, dot_product


def compute_legendre_integral_dot_product(basis_dimension):
    """
    Compute integrals and dot products of Legendre polynomials.

    Input:
        - basis_dimension: dict
            Give the number of basis functions for each state
    Outputs:
        - integral: ndarray
            Array containing the integrals over an interval
        - dot_product: ndarray
            Matrix containing the dot products
    """
    # Trajectories are formally defined on [0,1] so the interval length is 1
    duration = 1
    # Compute the dimension of the problem
    dimension = np.sum([basis_dimension[elt] for elt in basis_dimension])

    # For Legendre polynomials, integral = 0 except when k = 0
    integral = np.zeros(dimension)
    i = 0
    for state in basis_dimension:
        integral[i] += duration
        i += basis_dimension[state]

    # Compute dot product between the polynomials
    # Here use <P_i, P_j> = duration / (2*i + 1) * delta_il=j
    dot_product = np.zeros([dimension, dimension])
    i, j = 0, 0
    for state1 in basis_dimension:
        for state2 in basis_dimension:
            k_range = min([basis_dimension[state1],
                           basis_dimension[state2]])
            for k in range(k_range):
                # Squared L^2-norm of the k-th Legendre polynomial
                dot_product[i + k, j + k] = duration / (2*k + 1)
            j += basis_dimension[state2]
        j = 0
        i += basis_dimension[state1]

    return integral, dot_product


def compute_legendre_integral_derivatives(basis_dimension):
    """
    Compute integrals of Legendre polynomials derivatives.

    Input:
        - basis_dimension: dict
            Give the number of basis functions for each state
    Output:
        - integral_deriv: ndarray
            Array containing the integrals over an interval
    """
    integral_deriv = np.array([])
    for state in basis_dimension:
        # Use Legendre property: P(1) = 1, P(0)= 1 or -1
        x = [k % 2 * 2 for k in range(basis_dimension[state])]
        integral_deriv = np.append(integral_deriv, x)

    return integral_deriv


def compute_legendre_dot_product_derivatives(basis_dimension):
    """
    Compute dot products of Legendre polynomials and their derivatives.

    Input:
        - basis_dimension: dict
            Give the number of basis functions for each state
    Outputs:
        - dot_product_12: ndarray
            Array containing the dot products of Legendre polynomials
            with their derivatives
        - dot_product_22: ndarray
            Array containing the dot products of Legendre polynomials
            derivatives
    """
    # Compute the dimension of the problem
    dimension = np.sum([basis_dimension[elt] for elt in basis_dimension])
    dot_product_12 = np.zeros([dimension, dimension])
    dot_product_22 = np.zeros([dimension, dimension])
    i, j = 0, 0
    # Loop over states
    for state1 in basis_dimension:
        for state2 in basis_dimension:
            for k in range(basis_dimension[state1]):
                c_k = np.zeros(basis_dimension[state1])
                c_k[k] += 1
                # Create Legendre class for k-th polynomial
                c_k = Legendre(c_k, domain=[0, 1])
                # Compute derivative
                c_k_deriv = c_k.deriv()
                for l in range(basis_dimension[state2]):
                    c_l = np.zeros(basis_dimension[state2])
                    c_l[l] += 1
                    # Create Legendre class for k-th polynomial
                    c_l = Legendre(c_l, domain=[0, 1])
                    # Compute derivative
                    c_l_deriv = c_l.deriv()
                    # Multiply polynomials
                    product_12 = legmul(list(c_k), list(c_l_deriv))
                    product_22 = legmul(list(c_k_deriv), list(c_l_deriv))
                    # Create classes
                    product_12 = Legendre(product_12, domain=[0, 1])
                    product_22 = Legendre(product_22, domain=[0, 1])
                    # Integrate
                    int_product_12 = product_12.integ()
                    int_product_22 = product_22.integ()
                    # Evaluate at the endpoints
                    _, traj_deriv_12 = int_product_12.linspace(n=2)
                    _, traj_deriv_22 = int_product_22.linspace(n=2)
                    # Deduce dot products
                    dot_product_12[i + k, j + l] += traj_deriv_12[1]
                    dot_product_12[i + k, j + l] -= traj_deriv_12[0]
                    dot_product_22[i + k, j + l] += traj_deriv_22[1]
                    dot_product_22[i + k, j + l] -= traj_deriv_22[0]
            j += basis_dimension[state2]
        j = 0
        i += basis_dimension[state1]

    return dot_product_12, dot_product_22


def compute_bspline_integral_dot_product(basis_features, basis_dimension):
    """
    Compute integrals and dot products of B-splines.

    Input:
        - basis_features: dict
            Contain information on the basis for each state
        - basis_dimension: dict
            Give the number of basis functions for each state
    Outputs:
        - integral: ndarray
            Array containing the integrals over an interval
        - dot_product: ndarray
            Matrix containing the dot products
    """
    # Compute the dimension of the problem
    dimension = np.sum([basis_dimension[elt] for elt in basis_dimension])
    # Get the knots
    t = basis_features['knots']
    # FIXME: Consider small parameter to avoid vanishing of the last B-spline
    # at 1
    eps = 1e-16

    # Define integrals using antiderivatives
    integral = np.zeros(dimension)
    i0 = 0
    # Loop over states
    for state in basis_dimension:
        # Get degree of the B-splines of state
        k_state = basis_features[state]
        # Add external knots depending on the degree
        t_state = np.r_[(0,)*(k_state+1), t, (1,)*(k_state+1)]
        for i in range(basis_dimension[state]):
            # Define i-th B-spline
            spl_i = BSpline.basis_element(t_state[i:i+k_state+2])
            # Integrate and deduce integral
            spl_i_int = spl_i.antiderivative(nu=1)
            integral[i0+i] += spl_i_int(t_state[i+k_state+1] - eps)
            integral[i0+i] -= spl_i_int(t_state[i])
        i0 += basis_dimension[state]

    # Compute dot product between the B-splines
    dot_product = np.zeros([dimension, dimension])
    i, j = 0, 0
    # Loop over states
    for state1 in basis_dimension:
        # Get degree of the B-splines of state1
        k1 = basis_features[state1]
        # Add external knots depending on the degree
        t1 = np.r_[(0,)*(k1+1), t, (1,)*(k1+1)]
        for state2 in basis_dimension:
            # Get degree of the B-splines of state2
            k2 = basis_features[state2]
            # Add external knots depending on the degree
            t2 = np.r_[(0,)*(k2+1), t, (1,)*(k2+1)]
            for m in range(basis_dimension[state1]):
                # Define m-th B-spline of the state1 basis
                spl_m = BSpline.basis_element(t1[m:m+k1+2])
                for n in range(basis_dimension[state2]):
                    # Define n-th B-spline of the state2 basis
                    spl_n = BSpline.basis_element(t2[n:n+k2+2])
                    max_t = max(t1[m], t2[n])
                    min_t = min(t1[m+k1+1], t2[n+k2+1])
                    # If intersection of supports then do computations
                    if max_t < min_t:
                        # Numerical integration
                        quad_int = quad(lambda x: spl_m(x) * spl_n(x), max_t,
                                        min_t)
                        dot_product[i + m, j + n] += quad_int[0]
            j += basis_dimension[state2]
        j = 0
        i += basis_dimension[state1]

    return integral, dot_product


def compute_bspline_integral_derivatives(basis_dimension):
    """
    Compute integrals of B-splines.

    Input:
        - basis_dimension: dict
            Give the number of basis functions for each state
    Output:
        - integral_deriv: ndarray
            Array containing the integrals over an interval
    """
    # Compute the dimension of the problem
    dimension = np.sum([basis_dimension[elt] for elt in basis_dimension])

    # Fill using antiderivatives and the fact that only the first and last
    # splines are non-zero at the endpoints with value 1
    integral_deriv = np.zeros(dimension)
    i0 = 0
    # Loop over states
    for state in basis_dimension:
        integral_deriv[i0] += -1.
        integral_deriv[i0+basis_dimension[state]-1] += 1.
        i0 += basis_dimension[state]

    return integral_deriv


def compute_bspline_dot_product_derivatives(basis_features, basis_dimension):
    """
    Compute dot products of B-splines and their derivatives.

    Input:
        - basis_features: dict
            Contain information on the basis for each state
        - basis_dimension: dict
            Give the number of basis functions for each state
    Outputs:
        - dot_product_12: ndarray
            Array containing the dot products of Legendre polynomials
            with their derivatives
        - dot_product_22: ndarray
            Array containing the dot products of Legendre polynomials
            derivatives
    """
    # Compute the dimension of the problem
    dimension = np.sum([basis_dimension[elt] for elt in basis_dimension])
    # Get the knots
    t = basis_features['knots']
    # FIXME: Consider small parameter to avoid vanishing of the last B-spline
    # at 1
    eps = 1e-16

    dot_product_12 = np.zeros([dimension, dimension])
    dot_product_22 = np.zeros([dimension, dimension])
    i, j = 0, 0
    # Loop over states
    for state1 in basis_dimension:
        # Get degree of the B-splines of state1
        k1 = basis_features[state1]
        # Add external knots depending on the degree
        t1 = np.r_[(0,)*(k1+1), t, (1,)*(k1+1)]
        for state2 in basis_dimension:
            # Get degree of the B-splines of state2
            k2 = basis_features[state2]
            # Add external knots depending on the degree
            t2 = np.r_[(0,)*(k2+1), t, (1,)*(k2+1)]
            for m in range(basis_dimension[state1]):
                # Define m-th B-spline of the state1 basis
                spl_m = BSpline.basis_element(t1[m:m+k1+2])
                # Reproduce the same spline for differenciation because of
                # differenciation problems with BSpline.basis_element()
                # FIXME: simplify if possible
                # Construct knots by first finding the internal knots and then
                # by adding the right numbers of external knots
                t1m = t1[m:m+k1+2]
                ind_min1 = np.max(np.argwhere(t1m == t1[m]))
                ind_max1 = np.min(np.argwhere(t1m == t1[m+k1+1]))
                t_m = np.r_[(t1m[ind_min1],)*k1,
                            t1m[ind_min1:ind_max1+1],
                            (t1m[ind_max1],)*k1]
                x_m = np.linspace(t1m[0], t1m[-1]-eps, 50)
                spl_m = make_lsq_spline(x_m, spl_m(x_m), t_m, k1)
                # Compute derivative
                spl_m_deriv = spl_m.derivative(nu=1)
                for n in range(basis_dimension[state2]):
                    # Define n-th B-spline of the state2 basis
                    spl_n = BSpline.basis_element(t2[n:n+k2+2])
                    # FIXME: simplify if possible
                    # Construct knots by first finding the internal knots and
                    # then by adding the right numbers of external knots
                    t2n = t2[n:n+k2+2]
                    ind_min2 = np.max(np.argwhere(t2n == t2[n]))
                    ind_max2 = np.min(np.argwhere(t2n == t2[n+k2+1]))
                    t_n = np.r_[(t2n[ind_min2],)*k2,
                                t2n[ind_min2:ind_max2+1],
                                (t2n[ind_max2],)*k2]
                    x_n = np.linspace(t2n[0], t2n[-1]-eps, 50)
                    spl_n = make_lsq_spline(x_n, spl_n(x_n), t_n, k2)
                    # Compute derivative
                    spl_n_deriv = spl_n.derivative(nu=1)
                    max_t = max(t1[m], t2[n])
                    min_t = min(t1[m+k1+1], t2[n+k2+1])
                    # If intersection of supports then do computations
                    if max_t < min_t:
                        # Numerical integration
                        quad_int_12 = quad(lambda x:
                                           spl_m(x) * spl_n_deriv(x),
                                           max_t, min_t)
                        quad_int_22 = quad(lambda x:
                                           spl_m_deriv(x) * spl_n_deriv(x),
                                           max_t, min_t)
                        dot_product_12[i + m, j + n] += quad_int_12[0]
                        dot_product_22[i + m, j + n] += quad_int_22[0]
            j += basis_dimension[state2]
        j = 0
        i += basis_dimension[state1]

    return dot_product_12, dot_product_22
