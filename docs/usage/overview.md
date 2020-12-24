## Overview

PyRotor is a Python library for trajectory optimisation problems. It has been initially developed for the aeronautic setting but it is intended to be generic and to be used in a wide range of applications.

The problem of optimising a trajectory can be modelled via optimal control approaches for which numerous numerical methods exist. Nevertheless the differential system involved in the modelling may be (partially) unknown in certain cases. Different strategies, such as parameter estimation methods, have been designed to circumvent this problem but the resulting system contains necessarily some uncertainties. This may impact at different levels the solution of the optimisation problem.

In view of this, a different generic methodology has been proposed in this [paper](https://arxiv.org/abs/2011.11820) to obtain optimised and realistic trajectories without involving noisy and complex dynamical systems. This is achieved by using more directly the available data: the approach leads to optimisation problems which are simply and naturally constrained by the data. The resulting problem focuses the search on a region centered on the data and involves estimated constraints.

Currently PyRotor covers the case of cost functions defined via a quadratic instantaneous cost. Note that this instantaneous cost can be an estimated model resulting from a learning step. The optimisation process is performed in a finite-dimensional space onto which the trajectories have been projected. The first functional basis implemented in PyRotor is actually given by the Legendre basis. Future releases of PyRotor will cover more general instantaneous costs and bases.
At the same time, PyRotor solves optimisation problems with
* linear boundary constraints (with tolerated errors);
* path constraints (including bounds on the variables).

We refer to the [tutorial page](usage/tutorial.md) for an example on how PyRotor works. More examples will soon be available in the [example page](usage/features.md).
