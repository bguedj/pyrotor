import numpy as np


"""
Describe the iterative process performed while optimizing trajectories.
"""


def get_kappa_boundaries(reference_coefficients, q, w, sigma_inverse, c_weight):
    """
    Give the possible minumum and maximum supposed value of kappa.

    Inputs:
        - reference_coefficients: ndarray
            Coefficients of reference
        - q: ndarray
            Matrix of the quadratic term
        - w: ndarray
            Vector of the linear term (without intercept)
        - sigma_inverse: ndarray
            Pseudoinverse of the covariance matrix of the reference
            coefficients
        - c_weight: ndarray
            Coefficients of a weighted trajectory

    Outputs:
        - kappa_min: float
            Supposed possible minimum value of kappa
        - kappa_max: float
            Supposed possible maximum value of kappa

    """
    f = []
    g = []
    for reference_coefficient in reference_coefficients:
        f.append(compute_f(reference_coefficient, sigma_inverse, c_weight))
        g.append(compute_g(reference_coefficient, q, w))
    kappa_mean = compute_kappa_mean(f, g)

    kappa_min = compute_kappa_min(kappa_mean)
    kappa_max = compute_kappa_max(kappa_mean)

    return kappa_min, kappa_max


def compute_kappa_min(kappa_mean):
    """
    Compute the supposed possible minimum value of kappa

    Inputs:
        - kappa_mean: float
            Mean kappa

    Output:
        - kappa_min: float
            Supposed possible minimum value of kappa
    """
    return kappa_mean / 1000


def compute_kappa_max(kappa_mean):
    """
    Compute the supposed possible maximum value of kappa

        Inputs:
            - kappa_mean: float
                Mean kappa

        Output:
            - kappa_max: float
                Supposed possible maximum value of kappa
    """
    return kappa_mean * 1000


def compute_kappa_mean(f, g):
    """
    Compute the mean kappa

    Inputs:
        - f: list
            Evaluations of several reference coefficients over f
        - f: list
            Evaluations of several reference coefficients over g

    Output:
        kappa_mean: float
            Mean kappa
    """
    f = np.array(f)
    g = np.array(g)
    kappa = f/g
    return np.mean(kappa)


def compute_f(x, sigma_inverse, c_weight):
    """
    Evaluate the coefficients of a single trajectory over f. Where f is the
    cost function given by the user.

    Inputs:
        - x: ndarray
            Coefficients of a single trajectory.
        - sigma_inverse: ndarray
            Pseudoinverse of the covariance matrix of the reference
            coefficients.
        - c_weight: ndarray
            Coefficients of a weighted trajectory

    Output:
        - cost: float
            The cost of the given trajectory (by its coefficients) over the
            cost function given by the user.
    """
    a = np.dot(np.dot(x.T, sigma_inverse), x)
    b = np.dot(2 * np.dot(sigma_inverse, c_weight).T, x)
    return a - b


def compute_g(x, Q, W):
    """
    Evaluate the coefficients of a single trajectory over g. Where g is the
    function penalizing the distance between the optimized trajectory and the
    reference trajectories.

    Inputs:
        - x: ndarray
            Coefficients of a single trajectory.
        - Q: ndarray
            Matrix of the quadratic term.
        - W: ndarray
            Vector of the linear term (without intercept).

    Output:
        - g(x): float
            Evaluation of x over g.
    """
    a = np.dot(np.dot(x.T, Q), x)
    b = np.dot(W.T, x)
    return a + b


def iterate_through_kappas(trajectory, kappa_min, kappa_max):
    """
    Iterate through the different kappas in order to find the optimum
    trajectory that follows our constraints.

    Inputs:
        - trajectory: Pyrotor instance
            Trajectory to optimize.
        - kappa_min: float
            Supposed minimum possible kappa
        - kappa_max: float
            Supposed maximum possible kappa
    """
    trajectory.kappas = np.linspace(kappa_min, kappa_max, 1000)
    trajectory.i_binary_search = 0
    binary_search_best_trajectory(trajectory,
                                  len(trajectory.kappas)-1,
                                  len(trajectory.kappas)-1)
    if not self.is_valid:
        raise ValueError("Trajectories of reference too close to your constraints:\nAborted")


def binary_search_best_trajectory(trajectory, i, step):
    """
    Perfor a binary search amoung all the kappas to find the best trajectory

    Inputs:
        - i: int
            index of the kappa to use
        - step: int
            size of the current split
    """
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
