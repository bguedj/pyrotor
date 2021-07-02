## Tutorial

Here we explain how PyRotor works through the example given in the notebook [getting_started.ipynb](https://github.com/bguedj/pyrotor/tree/master/examples/getting_started.ipynb).

We consider a very simple problem in dimension 2. It consists in optimising a trajectory in the square [0,1]x[0,1] which is always above a forbidden region. The cost function models the distance of the curve to the origin.

First we need to import Pyrotor and Numpy to define our problem.

```python
import numpy as np
import pyrotor
```

Then we import the trajectories of reference from a toy dataset. Note that these trajectories have been generated using the notebook [generate.ipynb](https://github.com/bguedj/pyrotor/tree/master/examples/generate.ipynb).

```python
reference_trajectories = pyrotor.datasets.load_toy_dataset("example_1")
# Visualise the data
print(reference_trajectories[0].head())
```

The next step is to define the cost function. Currently PyRotor covers costs defined by the integral of a quadratic instantaneous cost. The user defines the matrix, the vector and the real number giving the quadratic instantaneous cost.
The user can also indicates a path to a quadratic model in pickle format, resulting from a learning process; see the [objective_matrices](core_module/objective_matrices.md) page.

```python
# Quadratic part
q = np.array([[0,0],
              [0,1]])
# Linear part
w = np.array([0,0])

# Constant part
c = 1

quadratic_model = [c, w, q]
```

Now we set the initial and final conditions for the two variables. Note that a tolerated error, modelled by the parameter 'delta', can be taken into account.

```python
endpoints = {'x1': {'start': .111,
                    'end': .912,
                    'delta': 0.0001},
             'x2': {'start': .926,
                    'end': .211,
                    'delta': 0.0001}}
```

The trajectories being parametrised, we define the independent variable (time).

```python
independent_variable = {'start': .1,
                        'end': .9,
                        'frequency': .01}
# Compute number of evaluation points
delta_time = independent_variable['end'] - independent_variable['start']
delta_time /= independent_variable['frequency']
independent_variable['points_nb'] = int(delta_time) + 1
```

As explained above, the trajectory should remain in the square and outside a forbidden region. These constraints are then defined as functions.

```python
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
```

The trajectories are projected onto a finite-dimension space for the optimisation. Here we define the basis and the dimension for each variable. Currently Legendre polynomials and B-splines are implemented but the methodology is not restricted to these two families.
Note that if the user considers B-splines then it has to define the position of the internal knots in the interval [0,1].

```python
basis = 'bspline'
if basis == 'legendre':
    basis_features = {'x1': 4,
                      'x2': 6}
elif basis == 'bspline':
    basis_features = {'knots': [.33, .66],
                      'x1': 3,
                      'x2': 4}
```

We create now an instance to model our problem. Note that the user can choose the number of reference trajectories and the value of the optimisation factor. This factor models the balance between optimising and staying close to reference trajectories: the larger, the more we optimise.
Through the argument 'sigma_inverse', the user can define manually the covariance matrix modelling the relations between each coefficient of the variables; this matrix is by default estimated from the reference trajectories (the default value is used in the present example). In this case, the solution tends to reproduce the pattern of the reference trajectories.
The user can also decide to use the quadratic programming solver 'qp' from [CVXOPT](http://cvxopt.org/userguide/coneprog.html#quadratic-programming). Otherwise the generic solver 'minimize' from [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) is used.

```python
mr_pyrotor = pyrotor.Pyrotor(quadratic_model,
                             reference_trajectories,
                             endpoints,
                             constraints,
                             basis,
                             basis_features,
                             independent_variable,
                             n_best_trajectory_to_use=5,
                             opti_factor=1,
                             use_quadratic_programming=True,
                             verbose=False)
```

Let us compute the optimised trajectory.

```python
mr_pyrotor.compute_optimal_trajectory()
```

We compute the savings to assess the optimisation.

```python
savings = pd.Series(mr_pyrotor.compute_gains(), name='Savings')
print(savings)
relative_savings = pd.Series(mr_pyrotor.compute_relative_gains() * 100, name='Relative savings [%]')
print(relative_savings)
```

And we finally plot the results.

```python
# Define time axis
X = np.linspace(independent_variable['start'],
                independent_variable['end'],
                independent_variable['points_nb'])

# Define nonlinear constraint to plot
X_ = np.linspace(0, 1, 101)
constraint_f5 = np.array([150/19 * (1-x)**3 - 225/19 * (1-x)**2 + 100/19 * (1-x) - 79/190 for x in X_])

# Plot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 20))
# Plot first variable with respect to time
ax1.plot(X, mr_pyrotor.trajectory['x1'])
ax1.set_xlabel('$t$')
ax1.set_ylabel('$x_1(t)$')
# Plot second variable with respect to time
ax2.plot(X, mr_pyrotor.trajectory['x2'])
ax2.set_xlabel('$t$')
ax2.set_ylabel('$x_2(t)$')
# Plot in (x_1, x_2) space
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
```
