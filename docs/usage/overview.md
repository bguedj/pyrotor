## Overview

Pyrotor compute an optimized trajectory thanks to real ones. This trajectory is a sequence of controls (expressed as a function) that moves the dynamical system between two points in state space. The trajectory will minimize some cost function, which is typically an integral along the trajectory. The trajectory will also satisfy a set user-defined constraints.

OptimTraj solves problems with

    continuous dynamics
    boundary constraints
    path constraints
    integral cost function
