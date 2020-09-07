
[![](https://travis-ci.com/bguedj/pyrotor.svg?token=mBozP3BYCpx6zxMpJQAQ&branch=master)](https://travis-ci.com/github/bguedj/pyrotor)

# pyrotor
*Trajectory optimization package based on data*

- - - -
### Installation

```Bash
$ pip install pyrotor
```

- - - -
### Getting started
```Python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
import pyrotor


# Define setting

# Define quadratic cost function
# Quadratic part
q = np.array([[1,0],
             [0,1]])
# Linear part
w = np.array([0,0])

# Constant part
c = 2.87

quadratic_model = [c, w, q]


# Define initial and final states
endpoints = {'x1': {'start': .1,
                    'end': .9,
                    'delta': .01},
             'x2': {'start': .828,
                    'end': .172,
                    'delta': .01}}
# Define independent variable (time)

independent_variable = {'start': .1,
                        'end': .9,
                        'frequency': .01}
# Compute number of evaluation points
delta_time = independent_variable['end'] - independent_variable['start']
delta_time /= independent_variable['frequency']
independent_variable['points_nb'] = int(delta_time) + 1


# Define constraints

# x1 > 0
def f1(data):
    x1 = data["x1"].values
    return x1

# x1 < 1
def f2(data):
    x1 = data["x1"].values
    return 1 - x1

# x2 > 0
def f3(data):
    x2 = data["x2"].values
    return x2

# x2 < 1
def f4(data):
    x2 = data["x2"].values
    return 1 - x2

# x2 > f(x1)
def f5(data):
    x1 = data["x1"].values
    x2 = data["x2"].values
    return x2 - 150/19 * (1-x1)**3 + 225/19 * (1-x1)**2 - 100/19 * (1-x1) + 79/190

constraints = [f1, f2, f3, f4, f5]


# Define functional basis

# Basis name
basis = 'legendre'
# Dimension for each variable
basis_dimension = {'x1': 3,
                   'x2': 5}

# Import reference trajectories

reference_trajectories = pyrotor.datasets.load_toy_dataset()

# Optimization

# Create PyRotor class

mr_pyrotor = pyrotor.Pyrotor(quadratic_model,
                             reference_trajectories,
                             endpoints,
                             constraints,
                             basis,
                             basis_dimension,
                             independent_variable,
                             n_best_trajectory_to_use=20,
                             verbose=False)


# Execute PyRotor solver

mr_pyrotor.compute_optimal_trajectory()


# Compute savings

mr_pyrotor.compute_gains()
mr_pyrotor.compute_relative_gains()


# Plot

X = np.linspace(independent_variable['start'],
                independent_variable['end'],
                independent_variable['points_nb'])
X_ = np.linspace(0, 1, 101)
constraint_f5 = np.array([150/19 * (1-x)**3 - 225/19 * (1-x)**2 + 100/19 * (1-x) - 79/190 for x in X_])

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 20))
ax1.plot(X, mr_pyrotor.trajectory['x1'])
ax1.set_xlabel('$t$')
ax1.set_ylabel('$x_1(t)$')
ax2.plot(X, mr_pyrotor.trajectory['x2'])
ax2.set_xlabel('$t$')
ax2.set_ylabel('$x_2(t)$')
ax3.plot(mr_pyrotor.trajectory['x1'], mr_pyrotor.trajectory['x2'], color='b', label='Optimized trajectory')
for trajectory in mr_pyrotor.reference_trajectories:
    ax3.plot(trajectory['x1'], trajectory['x2'], linestyle=":", label='_nolegend_')
ax3.fill_between(X_, 0, constraint_f5, color='r', alpha=.5, label='Forbidden area')
ax3.set_xlabel('$x_1$')
ax3.set_ylabel('$x_2$')
ax3.set_xlim(left=0, right=1)
ax3.set_ylim(bottom=0, top=1)
ax3.legend()
plt.tight_layout()
plt.show()
```
