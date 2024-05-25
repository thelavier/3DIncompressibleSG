import numpy as np

from geomloss import SamplesLoss
import torch

import auxfunctions as aux

def Wasserstein_Distance(Z, M, ZRef, MRef, indices):
    """
    Computes the Wasserstein distance (Sinkhorn divergence approximation) between two distributions for aligned timesteps.

    Args:
        Z (array): Current seed positions.
        M (array): Current mass distribution.
        ZRef (array): Reference seed positions.
        MRef (array): Reference mass distribution.
        indices (list): List of tuples with indices for comparison.

    Returns:
        list: List of Wasserstein distances at each comparison time.
    """
    use_cuda = torch.cuda.is_available()
    dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    Loss = SamplesLoss(
            "sinkhorn",
            p=2,
            debias=True, 
            backend="multiscale"
        )

    distances = []
    counter = 1 
    for ind, indRef in indices:
        a = torch.from_numpy(M[ind]).type(dtype)
        b = torch.from_numpy(MRef[indRef]).type(dtype)
        x = torch.from_numpy(Z[ind]).type(dtype)
        y = torch.from_numpy(ZRef[indRef]).type(dtype)

        sol = Loss(a, x, b, y)
        distances.append(sol.item())

        print(counter, 'of', len(indices))
        counter += 1

    return distances

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
    return sol * normalization
    
def Root_Mean_Squared_Velocity(Z, C, Type):
    """
    Computes the root mean squared velocity for given seed positions and types.

    Args:
        Z (array): Seed positions.
        C (array): Centroid positions.
        Type (str): Type of velocity to compute ('Meridional', 'Zonal', or 'Total').

    Returns:
        list: Root mean squared velocity for each timestep.
    """
    Vel = aux.get_velocity(Z, C, Type)

    results = []
    for i in range(len(Vel)):
        if Type in ['Meridional', 'Zonal']:
            results.append(np.sqrt(np.mean(np.abs(Vel[i]) ** 2)))
        elif Type == 'Total':
            results.append(np.sqrt(np.mean(np.linalg.norm(Vel[i], axis=1) ** 2)))

    return results