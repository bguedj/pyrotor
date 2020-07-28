import numpy as np


"""
Describe the iterative process performed while optimizing trajectories.
"""


def get_kappa_boundaries(x, Q, W, sigma_inverse, c_weight):
    # FIXME: loop through many ref trajectories
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


def iterate_through_kappas(trajectory, kappa_min, kappa_max):
    trajectory.kappas = np.linspace(kappa_min, kappa_max, 1000)
    trajectory.i_binary_search = 0
    binary_search_best_trajectory(trajectory,
                                  len(trajectory.kappas)-1,
                                  len(trajectory.kappas)-1)
    if not self.is_valid:
        raise ValueError("Trajectories of reference too close to your constraints:\nAborted")


def binary_search_best_trajectory(trajectory, i, step):
    trajectory.i_kappa = i
    trajectory.i_binary_search += 1
    if i < 0:
        raise ValueError("Trajectories of reference too close to your constraints:\nAborted")

    trajectory.kappa = trajectory.kappas[i]
    trajectory.compute_trajectory()

    step = step//2
    if not trajectory.is_valid:
        if step == 0:
            step = 1
        binary_search_best_trajectory(trajectory, i-step, step)
    else:
        if len(trajectory.kappas)-1 != i and step != 0:
            binary_search_best_trajectory(trajectory, i+step, step)
