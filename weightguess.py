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

    min_Z, max_Z = np.min(Z, axis=0), np.max(Z, axis=0)
    lambda_vals = []

    # Calculate lambda_ for each non-periodic dimension
    for i, periodic in enumerate([PeriodicX, PeriodicY, PeriodicZ]):
        if not periodic:
            min_b = bx[i]
            max_b = bx[i + 3]
            lambda_dim = (max_b - min_b) / (max_Z[i] - min_Z[i])
            lambda_vals.append(lambda_dim)

    # Choose the minimum lambda_ and apply a small reduction
    lambda_ = min(lambda_vals) * (1 - 1e-2) if lambda_vals else np.inf

    # Calculate translation vector for non-periodic dimensions
    translation = []
    for i, periodic in enumerate([PeriodicX, PeriodicY, PeriodicZ]):
        if not periodic:
            center_dom = (bx[i + 3] + bx[i]) / 2
            center_rescaled = lambda_ * (min_Z[i] + max_Z[i]) / 2
            translation.append(center_dom - center_rescaled)
        else:
            translation.append(0)
    t = np.array(translation)[~np.array([PeriodicX, PeriodicY, PeriodicZ])]

    # Calculate weights
    Z_mod = Z[:, ~np.array([PeriodicX, PeriodicY, PeriodicZ])]
    w = (1 - lambda_) * np.square(np.linalg.norm(Z_mod, axis=1)) - 2 * np.dot(Z_mod, t) - psi / lambda_

    return w, lambda_, t