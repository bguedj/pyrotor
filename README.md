
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
- Florent Dewez, Benjamin Guedj, Arthur Talpaert, Vincent Vandewalle. An end-to-end data-driven optimisation framework for constrained trajectories. Preprint [https://arxiv.org/abs/2011.11820](https://arxiv.org/abs/2011.11820).

Please consider citing the preprint if you are using the library:

```
@unpublished{dewez2020endtoend,
title={An end-to-end data-driven optimisation framework for constrained trajectories}, 
author={Florent Dewez and Benjamin Guedj and Arthur Talpaert and Vincent Vandewalle},
year={2020},
abstract = {Many real-world problems require to optimise trajectories under constraints. Classical approaches are based on optimal control methods but require an exact knowledge of the underlying dynamics, which could be challenging or even out of reach. In this paper, we leverage data-driven approaches to design a new end-to-end framework which is dynamics-free for optimised and realistic trajectories. We first decompose the trajectories on function basis, trading the initial infinite dimension problem on a multivariate functional space for a parameter optimisation problem. A maximum \emph{a posteriori} approach which incorporates information from data is used to obtain a new optimisation problem which is regularised. The penalised term focuses the search on a region centered on data and includes estimated linear constraints in the problem. We apply our data-driven approach to two settings in aeronautics and sailing routes optimisation, yielding commanding results. The developed approach has been implemented in the Python library PyRotor.},
url = "https://arxiv.org/abs/2011.11820",
url_Software = "https://github.com/bguedj/pyrotor",
eprint={2011.11820},
archivePrefix={arXiv},
primaryClass={stat.AP}
}
```
