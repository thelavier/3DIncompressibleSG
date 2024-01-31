import jax
import jax.numpy as jnp
import numpy as np

from ott.geometry import pointcloud
from ott.core.sinkhorn import sinkhorn

import auxfunctions as aux

@jax.jit
def solve_ott(a, b, x, y, epsilon):
    n = len(a)
    threshold = 0.01 / (n**0.33)
    geom = pointcloud.PointCloud(jnp.array(x), jnp.array(y), epsilon = epsilon)
    out = sinkhorn( geom, jnp.array(a), jnp.array(b),
        threshold=threshold,
        max_iterations=1000,
        norm_error=2,
        lse_mode=True,
    )
    # center dual variables to facilitate comparison
    f, g, _, _, _  = out
    f, g = f - np.mean(f), g + np.mean(f)
    return f, g

def Sqrt_Sinkhorn_Loss(Z, M, ZRef, MRef, epsilon, tf, comptime, box):

    if comptime > tf:
        raise ValueError('Please select a valid comparison time.')
    
    # Compute the index that corresponds to the comparison time
    Ndt = len(Z)
    NdtRef = len(ZRef)
    ind = int(round(( Ndt / tf ) * comptime))
    indRef = int(round(( NdtRef / tf ) * comptime))

    # Compute the normalization
    Lx = box[3] - box[0]
    Ly = box[4] - box[1]
    Lz = box[5] - box[2]
    normalization = 1 / np.sqrt(np.abs(Lx * Ly* Lz) * np.max(np.max(np.abs(ZRef[indRef]), axis=1)) ** 2)

    # Compute the square root of the sinkhorn loss

    f, g = solve_ott(M[ind], MRef[indRef], Z[ind], ZRef[indRef], epsilon)
    p1, p2 = solve_ott(M[ind], M[ind], Z[ind], Z[ind], epsilon)
    print(p1, p2)
    q1, q2 = solve_ott(MRef[indRef], MRef[indRef], ZRef[indRef], ZRef[indRef], epsilon)
    print(q1, q2)

    sol = np.sqrt(np.dot(M[ind], f - p1) + np.dot(MRef[indRef], g - q1))

    return sol / normalization

def Weighted_Euclidian_Error(Z, ZRef, MRef, tf, comptime, box):

    if comptime > tf:
        raise ValueError('Please select a valid comparison time.')
    
    # Compute the index that corresponds to the comparison time
    Ndt = len(Z)
    NdtRef = len(ZRef)
    ind = int(round(( Ndt / tf ) * comptime))
    indRef = int(round(( NdtRef / tf ) * comptime))

    # Compute the normalization
    Lx = box[3] - box[0]
    Ly = box[4] - box[1]
    Lz = box[5] - box[2]
    normalization = 1 / np.sqrt(np.abs(Lx * Ly* Lz) * np.max(np.max(np.abs(ZRef[indRef]), axis=1)) ** 2)

    # Check that Z and ZRef are the same shape
    if np.shape(Z[ind]) != np.shape(ZRef[indRef]):
        raise ValueError('Please provide a valid comparison')
    
    diff = np.linalg.norm(Z[ind].astype(float) - ZRef[indRef].astype(float), axis = 1) ** 2
    sol = np.sqrt(np.dot(MRef[indRef], diff))
    return sol / normalization
    
def Root_Mean_Squared_Velocity(Z, C, W, Type):

    results = []

    if Type == 'Meridional':
        Vel = np.array(aux.MerVel(Z, C, W))
    elif Type == 'Zonal':
        Vel = np.array(aux.ZonVel(Z, C, W))
    elif Type == 'Total':
        MVel = np.array(aux.MerVel(Z, C, W))
        ZVel = np.array(aux.ZonVel(Z, C, W))
        Vel = np.dstack((ZVel, MVel))
    else:
        raise ValueError('Please specify which velocity you want to investigate. The options are Meriodonal, Zonal, or Total')
    
    if Type == 'Meridional' or Type == 'Zonal':
        for i in range(len(Vel)):
            results.append(np.sqrt(np.mean(np.abs(Vel[i]) ** 2)))
    elif Type == 'Total':
        for i in range(len(Vel)):
            results.append(np.sqrt(np.mean(np.linalg.norm(Vel[i], axis = 1) ** 2)))        
    else:
        raise ValueError('Please specify which velocity you want to investigate. The options are Meriodonal, Zonal, or Total')
    
    return results