import jax
import jax.numpy as jnp
import numpy as np

from ott.geometry import pointcloud
from ott.core.sinkhorn import sinkhorn

import auxfunctions as aux

@jax.jit
def solve_ott(a, b, x, y, epsilon):
    """
    Solves the optimal transport problem using the Sinkhorn algorithm.

    Args:
        a (array): Source measure.
        b (array): Target measure.
        x (array): Source points.
        y (array): Target points.
        epsilon (float): Regularization parameter for the Sinkhorn algorithm.

    Returns:
        tuple: f and g, the dual variables of the Sinkhorn algorithm.
    """
    n = len(a)
    threshold = 0.01 / (n**0.33)  # Adaptive threshold based on the problem size
    geom = pointcloud.PointCloud(jnp.array(x), jnp.array(y), epsilon=epsilon)
    out = sinkhorn(geom, jnp.array(a), jnp.array(b),
                   threshold=threshold,
                   max_iterations=1000,
                   norm_error=2,
                   lse_mode=True)
    # Center dual variables to facilitate comparison
    f, g, _, _, _ = out
    f, g = f - jnp.mean(f), g + jnp.mean(f)
    return f, g

def Sqrt_Sinkhorn_Loss(Z, M, ZRef, MRef, epsilon, tf, comptime, box):
    """
    Computes the square root of the Sinkhorn loss between two distributions.

    Args:
        Z (array): Current seed positions.
        M (array): Current mass distribution.
        ZRef (array): Reference seed positions.
        MRef (array): Reference mass distribution.
        epsilon (float): Regularization parameter for Sinkhorn algorithm.
        tf (float): Final time for comparison.
        comptime (float): Specific time for comparison.
        box (list/tuple): Domain boundaries.

    Returns:
        float: Normalized square root of the Sinkhorn loss.
    """
    if comptime > tf:
        raise ValueError('Please select a valid comparison time.')

    ind, indRef = aux.get_comparison_indices(len(Z), len(ZRef), tf, comptime)
    normalization = aux.compute_normalization(box, ZRef[indRef])

    f, g = solve_ott(M[ind], MRef[indRef], Z[ind], ZRef[indRef], epsilon)
    p1, p2 = solve_ott(M[ind], M[ind], Z[ind], Z[ind], epsilon)
    q1, q2 = solve_ott(MRef[indRef], MRef[indRef], ZRef[indRef], ZRef[indRef], epsilon)

    sol = np.sqrt(np.dot(M[ind], f - p1) + np.dot(MRef[indRef], g - q1))
    return sol / normalization

def Weighted_Euclidian_Error(Z, ZRef, MRef, tf, comptime, box):
    """
    Computes the weighted Euclidean error between two distributions.

    Args:
        Z (array): Current seed positions.
        ZRef (array): Reference seed positions.
        MRef (array): Reference mass distribution.
        tf (float): Final time for comparison.
        comptime (float): Specific time for comparison.
        box (list/tuple): Domain boundaries.

    Returns:
        float: Normalized weighted Euclidean error.
    """
    if comptime > tf:
        raise ValueError('Please select a valid comparison time.')

    ind, indRef = aux.get_comparison_indices(len(Z), len(ZRef), tf, comptime)
    normalization = aux.compute_normalization(box, ZRef[indRef])

    if Z[ind].shape != ZRef[indRef].shape:
        raise ValueError('Please provide a valid comparison.')

    diff = np.linalg.norm(Z[ind].astype(float) - ZRef[indRef].astype(float), axis=1) ** 2
    sol = np.sqrt(np.dot(MRef[indRef], diff))
    return sol / normalization
    
def Root_Mean_Squared_Velocity(Z, C, W, Type):
    """
    Computes the root mean squared velocity for given seed positions and types.

    Args:
        Z (array): Seed positions.
        C (array): Centroid positions.
        W (array): Weights/masses of the cells.
        Type (str): Type of velocity to compute ('Meridional', 'Zonal', or 'Total').

    Returns:
        list: Root mean squared velocity for each timestep.
    """
    Vel = aux.get_velocity(Z, C, W, Type)

    results = []
    for i in range(len(Vel)):
        if Type in ['Meridional', 'Zonal']:
            results.append(np.sqrt(np.mean(np.abs(Vel[i]) ** 2)))
        elif Type == 'Total':
            results.append(np.sqrt(np.mean(np.linalg.norm(Vel[i], axis=1) ** 2)))

    return results