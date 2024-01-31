import numpy as np
import msgpack

def load_data(data):
    """
    Loads data from a MessagePack file and returns numpy arrays of seeds, centroids, weights, and mass.

    Parameters:
        data (str): Filename of the MessagePack file to load.

    Returns:
        tuple: Four numpy arrays containing seeds, centroids, weights, and mass data.
    """
    # Initialize lists for data
    seeds, centroids, weights, mass = [], [], [], []

    # Load data from file
    with open(data, mode='rb') as msgpackfile:
        unpacker = msgpack.Unpacker(msgpackfile, raw=False)
        for row in unpacker:
            seeds.append(np.array(row.get('Seeds', []), dtype=np.float64))
            centroids.append(np.array(row.get('Centroids', []), dtype=np.float64))
            weights.append(np.array(row.get('Weights', []), dtype=np.float64))
            mass.append(np.array(row.get('Mass', []), dtype=np.float64))

    # Convert lists to numpy arrays and exclude the first entry
    Z, C, W, M = map(lambda x: np.array(x[1:]), (seeds, centroids, weights, mass))

    return Z, C, W, M


def get_remapped_seeds(box, Z, PeriodicX, PeriodicY, PeriodicZ):
    """
    Remaps seed positions to stay within a periodic domain.

    Parameters:
        box (list or tuple): Domain boundaries [xmin, ymin, zmin, xmax, ymax, zmax].
        Z (numpy.ndarray): Seed positions.
        PeriodicX (bool): Periodicity in the x-axis.
        PeriodicY (bool): Periodicity in the y-axis.
        PeriodicZ (bool): Periodicity in the z-axis.

    Returns:
        numpy.ndarray: Remapped seed positions.
    """
    if PeriodicX:
        Z[:, 0] = (Z[:, 0] - box[0]) % (box[3] - box[0]) + box[0]
    if PeriodicY:
        Z[:, 1] = (Z[:, 1] - box[1]) % (box[4] - box[1]) + box[1]
    if PeriodicZ:
        Z[:, 2] = (Z[:, 2] - box[2]) % (box[5] - box[2]) + box[2]

    return Z

def get_point_transform(point, matrix):
    """
    Transforms a point using a given transformation matrix.

    Parameters:
        point (list or numpy.ndarray): The point to transform.
        matrix (numpy.ndarray): Transformation matrix.

    Returns:
        list: Transformed point.
    """
    # Convert point to a 1x3 matrix and perform transformation
    point_matrix = np.array(point).reshape(3, 1)
    transformed_point = np.dot(matrix, point_matrix).flatten().tolist()

    return transformed_point

def Properties(Z, C, m, box):
    """
    Computes various physical properties based on seed and centroid positions.

    Parameters:
        Z (numpy.ndarray): Seed positions.
        C (numpy.ndarray): Centroid positions.
        m (numpy.ndarray): Mass array.
        box (list or tuple): Domain boundaries.

    Returns:
        tuple: Calculated Meridional Velocities, Zonal Velocities, Temperature, Total Energy, and Conservation Error.
    """
    # Parameters
    f, th0, g = 1, 1 ,1

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

def get_comparison_indices(Ndt, NdtRef, tf, comptime):
    """
    Computes the indices for comparison at a specific time from two time series.

    This function is used to find the corresponding indices in two time series datasets
    (e.g., positions or properties of particles) that are to be compared at a specific time.

    Args:
        Ndt (int): The number of data points in the first time series.
        NdtRef (int): The number of data points in the second time series.
        tf (float): The final time up to which the data points are recorded.
        comptime (float): The specific time at which the comparison is to be made.

    Returns:
        tuple: A tuple of two integers representing the indices in the first and second 
               time series respectively, corresponding to the comparison time.
    """
    ind = int(round((Ndt / tf) * comptime))
    indRef = int(round((NdtRef / tf) * comptime))
    return ind, indRef

def compute_normalization(box, ZRef):
    """
    Computes a normalization factor based on the domain size and reference positions.

    The normalization factor is used to scale certain calculations, such as error measures,
    to account for the size of the domain and the scale of the reference positions.

    Args:
        box (list/tuple): Domain boundaries [xmin, ymin, zmin, xmax, ymax, zmax].
        ZRef (array): Reference positions, typically a 2D numpy array where each row 
                      represents a position in space.

    Returns:
        float: A normalization factor based on the domain size and the maximum position
               magnitude in the reference positions.
    """
    Lx, Ly, Lz = box[3] - box[0], box[4] - box[1], box[5] - box[2]
    return 1 / np.sqrt(np.abs(Lx * Ly * Lz) * np.max(np.max(np.abs(ZRef), axis=1)) ** 2)

def get_velocity(Z, C, W, Type):
    """
    Calculate velocity components based on seed and centroid positions.

    Parameters:
        Z (numpy.ndarray): Seed positions.
        C (numpy.ndarray): Centroid positions.
        W (numpy.ndarray): Mass array.
        Type (str): Type of velocity to calculate ('Meridional', 'Zonal', or 'Total').

    Returns:
        numpy.ndarray: Velocity components based on the specified type.
            - If Type is 'Meridional', returns Meridional velocities.
            - If Type is 'Zonal', returns Zonal velocities.
            - If Type is 'Total', returns a stacked array of Zonal and Meridional velocities.
    
    Raises:
        ValueError: If an invalid velocity type is provided.
    """
    if Type == 'Meridional':
        return Z[:, :, 0] - C[:, :, 0]
    elif Type == 'Zonal':
        return C[:, :, 1] - Z[:, :, 1]
    elif Type == 'Total':
        MVel = Z[:, :, 0] - C[:, :, 0]
        ZVel = C[:, :, 1] - Z[:, :, 1]
        return np.dstack((ZVel, MVel))
    else:
        raise ValueError('Invalid velocity type. Use "Meridional", "Zonal", or "Total".')