import numpy as np


"""
Describe the iterative process performed while optimizing trajectories.
"""


def get_kappa_boundaries(x, Q, W, sigma_inverse, c_weight):
    f_0 = compute_f(x, sigma_inverse, c_weight)
    g_0 = compute_g(x, Q, W)

    kappa_mean = compute_kappa_mean(f_0, g_0)
    kappa_min = compute_kappa_min(kappa_mean)
    kappa_max = compute_kappa_max(kappa_mean)

    return kappa_min, kappa_max


def compute_kappa_min(kappa_mean):
    return kappa_mean / 1000


def compute_kappa_max(kappa_mean):
    return kappa_mean * 1000


def compute_kappa_mean(f_0, g_0):
    return f_0/g_0


def compute_f(x, sigma_inverse, c_weight):
    a = np.dot(np.dot(x.T, sigma_inverse), x)
    b = np.dot(2 * np.dot(sigma_inverse, c_weight).T, x)
    return a - b


def compute_g(x, Q, W):
    a = np.dot(np.dot(x.T, Q), x)
    b = np.dot(W.T, x)
    return a + b


def binary_search_best_trajectory(trajectory, i, step):
    trajectory.i_kappa = i
    trajectory.i_binary_search += 1
    if i < 0:
        raise ValueError("Trajectories of reference too close to your constraints:\nAborted")
    trajectory.weight_dict = {k: trajectory.kappas[i] * trajectory.original_weights[k]
                              for k in trajectory.original_weights}
    trajectory.compute_trajectory()

    step = step//2
    if not trajectory.is_valid:
        if step == 0:
            step = 1
        trajectory.binary_search_best_opti(i-step, step)
    else:
        if len(trajectory.kappas)-1 != i and step != 0:
            trajectory.binary_search_best_opti(i+step, step)


def compute_trajectory_kappa(trajectory):
    from copy import copy
    trajectory.original_weights = copy(trajectory.weight_dict)
    x = np.linspace(0, 1, 30)
    trajectory.kappas = 1/np.exp(5*x**(1/2))
    trajectory.i_binary_search = 0
    trajectory.binary_search_best_opti(len(trajectory.kappas)-1, len(trajectory.kappas)-1)
    if not self.is_valid:
        raise ValueError("Trajectories of reference too close to your constraints:\nAborted")
