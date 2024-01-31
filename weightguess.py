import numpy as np

def rescale_weights(bx, Z, psi, PeriodicX, PeriodicY, PeriodicZ):
    """
    Rescales weights for 3D Laguerre tessellation, ensuring positive cell areas.

    Parameters:
        bx (list/tuple): Domain [xmin, ymin, zmin, xmax, ymax, zmax].
        Z (numpy.ndarray): Seed points (shape: n x 3).
        psi (float): Initial guess for weights.
        PeriodicX, PeriodicY, PeriodicZ (bool): Periodicity flags for each axis.

    Returns:
        tuple: Weights (numpy.ndarray), scaling factor (float), and translation vector (numpy.ndarray).
    """
    if PeriodicX and PeriodicY and PeriodicZ:
        return psi, 0, 0

    # Initialize variables
    min_Z, max_Z = np.min(Z, axis=0), np.max(Z, axis=0)
    lambda_ = np.inf
    translation = []

    # Calculate scaling and translation for non-periodic dimensions
    for i, (periodic, min_b, max_b) in enumerate(zip([PeriodicX, PeriodicY, PeriodicZ], bx[::2], bx[1::2])):
        if not periodic:
            lambda_dim = (max_b - min_b) / (max_Z[i] - min_Z[i])
            lambda_ = min(lambda_, lambda_dim * (1 - 1e-2))
            center_dom = (max_b + min_b) / 2
            center_rescaled = lambda_ * (min_Z[i] + max_Z[i]) / 2
            translation.append(center_dom - center_rescaled)
        else:
            translation.append(0)

    t = np.array(translation)

    # Define weights
    Z_mod = Z[:, ~np.array([PeriodicX, PeriodicY, PeriodicZ])]
    w = (1 - lambda_) * np.square(np.linalg.norm(Z_mod, axis=1)) - 2 * np.dot(Z_mod, t) - psi / lambda_

    return w, lambda_, t