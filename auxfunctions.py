import numpy as np
import msgpack

def load_data(data):
    # Initialize lists to store the loaded data
    seeds = []
    centroids = []
    weights = []
    mass = []

    # Load the data from the MessagePack file
    with open(data, mode='rb') as msgpackfile:

        # Load the remaining data
        unpacker = msgpack.Unpacker(msgpackfile, raw=False)
        for row in unpacker:
            seeds.append(np.array(row.get('Seeds', []), dtype=object).astype(np.float64))
            centroids.append(np.array(row.get('Centroids', []), dtype=object).astype(np.float64))
            weights.append(np.array(row.get('Weights', []), dtype=object).astype(np.float64))
            mass.append(np.array(row.get('Mass', []), dtype=object).astype(np.float64))

    # Exclude the first entry from each list
    seeds = seeds[1:]
    centroids = centroids[1:]
    weights = weights[1:]
    mass = mass[1:]

    # Access the individual arrays
    Z = np.array(seeds)
    C = np.array(centroids)
    W = np.array(weights)
    M = np.array(mass)

    return Z, C, W, M


def get_remapped_seeds(box, Z, PeriodicX, PeriodicY, PeriodicZ):
    """
    A function that remaps the seeds so that they remain in the periodic domain

    Inputs:
        box: the fluid domain given as a list [xmin, ymin, zmin, x max, ymax, zmax]
        Z: the seed positions
        PeriodicX: a boolian specifying periodicity in x
        PeriodicY: a boolian specifying periodicity in y
        PeriodicZ: a boolian specifying periodicity in z

    Outputs:
        Z: the seeds remaped to be inside the fluid domain
    """
    
    if PeriodicX and PeriodicY and PeriodicZ:
        # Wrap points in x, y, and z components
        Z[:, 0] = (Z[:, 0] - box[0]) % (box[3] - box[0]) + box[0]
        Z[:, 1] = (Z[:, 1] - box[1]) % (box[4] - box[1]) + box[1]
        Z[:, 2] = (Z[:, 2] - box[2]) % (box[5] - box[2]) + box[2]
    elif PeriodicX and PeriodicY and not PeriodicZ:
        # Wrap points in the x and y component
        Z[:, 0] = (Z[:, 0] - box[0]) % (box[3] - box[0]) + box[0]
        Z[:, 1] = (Z[:, 1] - box[1]) % (box[4] - box[1]) + box[1]
    elif PeriodicX and not PeriodicY and PeriodicZ:
        # Wrap points in the x and z component
        Z[:, 0] = (Z[:, 0] - box[0]) % (box[3] - box[0]) + box[0]
        Z[:, 2] = (Z[:, 2] - box[2]) % (box[5] - box[2]) + box[2]
    elif not PeriodicX and PeriodicY and PeriodicZ:
        # Wrap points in the y and z component
        Z[:, 1] = (Z[:, 1] - box[1]) % (box[4] - box[1]) + box[1]
        Z[:, 2] = (Z[:, 2] - box[2]) % (box[5] - box[2]) + box[2]
    elif PeriodicX and not PeriodicY and not PeriodicZ:
        # Wrap points in the x component
        Z[:, 0] = (Z[:, 0] - box[0]) % (box[3] - box[0]) + box[0]
    elif not PeriodicX and PeriodicY and not PeriodicZ:
        # Wrap points in the y component
        Z[:, 1] = (Z[:, 1] - box[1]) % (box[4] - box[1]) + box[1]
    elif not PeriodicX and not PeriodicY and PeriodicZ:
        # Wrap points in the z component
        Z[:, 2] = (Z[:, 2] - box[2]) % (box[5] - box[2]) + box[2]
    
    return Z

def get_point_transform(point, matrix):
    """
    Inputs:
        point: 
        matrix:

    Outputs:
        transformed_point:
    """
    # Convert corner to a 1x3 matrix
    point_matrix = np.array(point).reshape(3, 1)
    
    # Perform the transformation
    transformed_point0 = np.dot(matrix, point_matrix)
    
    # Convert back to a list
    transformed_point = transformed_point0.flatten().tolist()
    
    return transformed_point

def Properties(Z, C, m, box):
    # Parameters
    f = 1
    th0 = 1
    g = 1

    # Compute Meridonal Velocities
    MVel = f * (Z[:, :, 0] - C[:, :, 0])

    # Compute Zonal Velocities
    ZVel = f * (C[:, :, 1] - Z[:, :, 1])

    # Compute Temperature
    T = (th0 * f ** 2) / g * Z[:, :, 2]

    # Compute Integral of |x|^2
    domvol = (1 / 3) * (box[0] - box[3]) * (box[1] - box[4]) * (box[2] - box[5]) * \
       (box[0] ** 2 + box[1] ** 2 + box[2] ** 2 + box[3] ** 2 + box[4] ** 2 + box[5] ** 2 + box[0] * box[3] + box[1] * box[4] + box[2] * box[5])

    # Vectorized computation of Kinetic Energy for each point at every timestep
    norm_Z_squared = np.sum(Z.astype(float) ** 2, axis=2)
    dot_Z_C = np.sum(Z * C, axis=2)
    
    totalEnergy = (f ** 2 / 2) * (domvol + np.sum(m * norm_Z_squared, axis=1) - 2 * np.sum(m * dot_Z_C, axis=1)) 
    
    meanEnergy = np.mean(totalEnergy)

    ConservationError = (meanEnergy - totalEnergy) / meanEnergy

    return MVel, ZVel, T, totalEnergy, ConservationError