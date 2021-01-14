
[![](https://travis-ci.com/bguedj/pyrotor.svg?token=mBozP3BYCpx6zxMpJQAQ&branch=master)](https://travis-ci.com/github/bguedj/pyrotor)

# PyRotor - Python Route Trajectory Optimiser
PyRotor is a Python library for trajectory optimisation problems. Initially developed for the aeronautic setting, it is intended to be generic and to be used in a wide range of applications.

PyRotor leverages available trajectory data to focus the search space and to estimate some properties which are then incorporated in the optimisation problem. This constraints in a natural and simple way the optimisation problem whose solution inherits realistic patterns from the data. In particular it does not require any knowledge on the dynamics of the system.

- - - -
### Documentation and examples

An online documentation is available [here](https://pyrotor.readthedocs.io/en/latest/) but is still under construction. Thereotical details on the methodology are available in the below reference.

The [examples](https://github.com/bguedj/pyrotor/tree/master/examples) folder provides currently two examples showing the use of PyRotor.

- - - -
### Installation
Run the following command:
```Bash
$ pip install pyrotor
```

- - - -
### Reference
- Florent Dewez, Benjamin Guedj, Arthur Talpaert, Vincent Vandewalle. An end-to-end data-driven optimisation framework for constrained trajectories. [Preprint version](https://arxiv.org/abs/2011.11820).
