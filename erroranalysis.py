import sys
import jax
import jax.numpy as jnp
import numpy as np

from ott.geometry import pointcloud
from ott.core.sinkhorn import sinkhorn

import csv
csv.field_size_limit(min(sys.maxsize, 2147483646))

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

def load_data(data):

    # Initialize lists to store the loaded data
    seeds = []
    centroids = []
    weights = []
    mass = []

    # Load the data from the CSV file
    with open(data, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            seeds.append(eval(row['Seeds']))
            centroids.append(eval(row['Centroids']))
            weights.append(eval(row['Weights']))
            mass.append(eval(row['Mass']))

    # Access the individual arrays
    Z = np.array(seeds)
    C = np.array(centroids)
    W = np.array(weights)
    M = np.array(mass)

    return Z, C, W, M

def MerVel(Z, C, W):

    # Physical Constants
    f = 1e-4

    # Compute Meridonal Velocities
    MVel = [[0] * len(W[0]) for _ in range(len(W))]
    for i in range(len(W)):
        for j in range(len(W[0])):
            MVel[i][j] = f * (Z[i][j][0] - C[i][j][0])

    return MVel

def ZonVel(Z, C, W):

    # Physical Constants
    f = 1e-4

    # Compute Zonal Velocities
    ZVel = [[0] * len(W[0]) for _ in range(len(W))]
    for i in range(len(W)):
        for j in range(len(W[0])):
            ZVel[i][j] = f * ( - Z[i][j][1] + C[i][j][1])
        
    return ZVel

def Sqrt_Sinkhorn_Loss(Z, M, ZRef, MRef, epsilon, tf, comptime):

    if comptime > tf:
        raise ValueError('Please select a valid comparison time.')
    
    # Compute the index that corresponds to the comparison time
    Ndt = len(Z)
    NdtRef = len(ZRef)
    ind = int(round(( Ndt / tf ) * comptime))
    indRef = int(round(( NdtRef / tf ) * comptime))

    # Compute the square root of the sinkhorn loss

    f, g = solve_ott(M[ind], MRef[indRef], Z[ind], ZRef[indRef], epsilon)
    p1, p2 = solve_ott(M[ind], M[ind], Z[ind], Z[ind], epsilon)
    print(p1, p2)
    q1, q2 = solve_ott(MRef[indRef], MRef[indRef], ZRef[indRef], ZRef[indRef], epsilon)
    print(q1, q2)

    sol = np.sqrt(np.dot(M[ind], f - p1) + np.dot(MRef[indRef], g - q1))

    return sol

def Weighted_Euclidian_Error(Z, ZRef, MRef, tf, comptime):

    if comptime > tf:
        raise ValueError('Please select a valid comparison time.')
    
    # Compute the index that corresponds to the comparison time
    Ndt = len(Z)
    NdtRef = len(ZRef)
    ind = int(round(( Ndt / tf ) * comptime))
    indRef = int(round(( NdtRef / tf ) * comptime))

    # Check that Z and ZRef are the same shape
    if np.shape(Z[ind]) == np.shape(ZRef[indRef]):
        diff = np.linalg.norm(Z[ind] - ZRef[indRef], axis = 1) ** 2
        sol = np.sqrt(np.dot(MRef[indRef], diff))
        return sol
    else:
        raise ValueError('Please provide a valid comparison')
    
def Root_Mean_Squared_Velocity(Z, C, W, Type):

    results = []

    if Type == 'Meridional':
        Vel = np.array(MerVel(Z, C, W))
    elif Type == 'Zonal':
        Vel = np.array(ZonVel(Z, C, W))
    elif Type == 'Total':
        MVel = np.array(MerVel(Z, C, W))
        ZVel = np.array(ZonVel(Z, C, W))
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