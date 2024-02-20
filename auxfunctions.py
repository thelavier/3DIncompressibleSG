import numpy as np
import msgpack
from scipy.integrate import nquad

def load_data(data):
    """
    Loads data from a MessagePack file and returns numpy arrays of seeds, centroids, weights, and mass.

    Parameters:
        data (str): Filename of the MessagePack file to load.

    Returns:
        tuple: Four numpy arrays containing seeds, centroids, weights, and mass data.
    """
    # Initialize lists for data
    seeds, centroids, weights, mass, transportcost = [], [], [], [], []

    # Load data from file
    with open(data, mode='rb') as msgpackfile:
        unpacker = msgpack.Unpacker(msgpackfile, raw=False)
        for row in unpacker:
            seeds.append(np.array(row.get('Seeds', []), dtype=np.float64))
            centroids.append(np.array(row.get('Centroids', []), dtype=np.float64))
            weights.append(np.array(row.get('Weights', []), dtype=np.float64))
            mass.append(np.array(row.get('Mass', []), dtype=np.float64))
            transportcost.append(np.array(row.get('TransportCost', []), dtype=np.float64))

    # Convert lists to numpy arrays and exclude the first entry
    Z, C, W, M, TC = map(lambda x: np.array(x[1:]), (seeds, centroids, weights, mass, transportcost))

    return Z, C, W, M, TC


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

def Properties(Z, C, TC):
    """
    Computes various physical properties based on seed and centroid positions.

    Parameters:
        Z (numpy.ndarray): Seed positions.
        C (numpy.ndarray): Centroid positions.
        TC (numpy.ndarray): Transport cost array.

    Returns:
        tuple: Calculated Meridional Velocities, Zonal Velocities, Temperature, Total Energy, and Conservation Error.
    """
    # Compute Meridonal Velocities
    MVel = (Z[:, :, 0] - C[:, :, 0])

    # Compute Zonal Velocities
    ZVel = (C[:, :, 1] - Z[:, :, 1])

    # Compute Temperature
    T = Z[:, :, 2]
    
    totalEnergy = np.sum(TC, axis=1)
    
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

def get_velocity(Z, C, Type):
    """
    Calculate velocity components based on seed and centroid positions.

    Parameters:
        Z (numpy.ndarray): Seed positions.
        C (numpy.ndarray): Centroid positions.
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
    
def cyc_perturb(x, y):
    """
    Calculate the cyclonic perturbation based on x and y coordinates.

    This function defines the perturbation that creates a cyclone by 
    modeling the cyclonic effect as a function of spatial coordinates.

    Parameters:
        x (float or np.ndarray): The x-coordinate(s) of the point(s).
        y (float or np.ndarray): The y-coordinate(s) of the point(s).

    Returns:
        float or np.ndarray: The cyclonic perturbation at the given coordinate(s).

    Note:
        The function is designed to work with both scalar and array inputs.
    """
    return (1 + (x/0.5)**2 + (y/0.5)**2)**(-(3/2)) - (1/2) * ((1 + ((x - 1)/0.5)**2 + (y/0.5)**2)**(-(3/2)) + (1 + ((x + 1)/0.5)**2 + (y/0.5)**2)**(-(3/2)))

def basic_state(x, y, z, A):
    """
    Define the basic state subject to perturbation based on spatial coordinates and a parameter.

    This function calculates the basic state of the system that is subsequently perturbed
    to model cyclonic activity. It incorporates a shear wind factor through the parameter A.

    Parameters:
        x (float or np.ndarray): The x-coordinate(s) of the point(s).
        y (float or np.ndarray): The y-coordinate(s) of the point(s).
        z (float or np.ndarray): The z-coordinate(s) of the point(s), representing height or depth.
        A (float): Shear wind factor influencing the basic state's perturbation.

    Returns:
        float or np.ndarray: The basic state at the given coordinate(s).

    Note:
        The function supports both scalar and array inputs for coordinates.
    """
    return 0.5 * (np.arctan2(y, 1 + z) - np.arctan2(y, 1 - z)) - 0.12 * y * z - 0.5 * A * (y**2 - z**2)

def grad_basic_state(x, y, z, A):
    """
    Compute the gradient of the basic state function at a given point.

    This function calculates the spatial gradient of the basic state, which is used
    to place seeds in the geostrophic space. It is essential for understanding how
    perturbations in the basic state might evolve.

    Parameters:
        x (float): The x-coordinate of the point. Note: The gradient with respect to x is always 0.
        y (float): The y-coordinate of the point.
        z (float): The z-coordinate (height or depth) of the point.
        A (float): Shear wind factor influencing the gradient computation.

    Returns:
        np.ndarray: A 3-element array representing the gradient of the basic state at the given point,
                    corresponding to [du/dx, du/dy, du/dz]. The gradient with respect to x is always 0,
                    reflecting the assumption of symmetry or lack of change in the x direction.

    Note:
        This function is designed for scalar inputs, reflecting its use for point-specific calculations.
    """
    grad = np.zeros(3, dtype=np.float64)

    # Calculate df_dy and df_dz using the derived formulas
    grad[0] = 0
    grad[1] = -A * y - 0.12 * z - 0.5 * (1 - z) / (y**2 + (1 - z)**2) + 0.5 * (z + 1) / (y**2 + (z + 1)**2)
    grad[2] = A * z - 0.12 * y - 0.5 * y / (y**2 + (z + 1)**2) - 0.5 * y / (y**2 + (1 - z)**2)

    return grad

def compute_fft_coefficients(box, truncation):
    """
    Compute FFT coefficients for cyclone perturbation functions.

    Parameters:
        box (list): The domain of the model specified as [xmin, ymin, zmin, xmax, ymax, zmax].
        truncation (int): The truncation level for Fourier series coefficients.

    Returns:
        numpy.ndarray: An array of combined Fourier series coefficients.
    """
    # Solve Laplace's equation for the perturbation
    a, b = box[3], box[4]

    Nx = 256  # Number of sample points in x
    Ny = 256  # Number of sample points in y

    # Boundary functions for solvability criterion
    def integrand_bottom(x, y):
        """Perturbation function at the bottom boundary."""
        return 0.15 * cyc_perturb(x, y)

    def integrand_top(x, y):
        """Perturbation function at the top boundary."""
        return -0.6 * cyc_perturb(x + 1, y)

    # Integrate boundary functions 
    If, _ = nquad(integrand_bottom, [[-a, a], [-b, b]])
    Ig, _ = nquad(integrand_top, [[-a, a], [-b, b]])

    # Create a grid and evaluate the perturbation functions
    x, y = np.linspace(-a, a, Nx), np.linspace(-b, b, Ny)
    X, Y = np.meshgrid(x, y)

    # Evaluate the function on the grid and subtract the average of the perturbations to enforce the solvability criterion
    F = 0.15 * cyc_perturb(X, Y) - If / (4 * a * b) 
    G = -0.6 * cyc_perturb(X + 1, Y) - Ig / (4 * a * b) 

    # Apply FFT
    F_fft = np.fft.rfft2(F) # Take the Fourier transform of the function at the bottom
    G_fft = np.fft.rfft2(G) # Take the Fourier transform of the function at the top

    # Initialize lists to store coefficients
    Fourier_coefficients = []

    # Reconstruct using the specified number of coefficients
    for kx in range(-truncation, truncation + 1):
        for ky in range(truncation + 1):  # Keeping ky positive as rfft2's last axis is half-sized
            # Calculate the correct indices for kx
            kx_index = (kx + Nx) % Nx
            coeff1 = 2 * F_fft[kx_index, ky] / (Nx * Ny)
            coeff2 = 2 * G_fft[kx_index, ky] / (Nx * Ny)
            # Store coefficients
            Fourier_coefficients.append([kx, ky, coeff1, coeff2])

    return Fourier_coefficients

def compute_kz(kx, ky, a, b):
    """
    Compute the wavenumber for given Fourier mode indices and domain sizes.

    This function calculates the wavenumber kz based on the indices of the
    Fourier modes (kx, ky) and the physical domain sizes (a, b) in the x and
    y directions, respectively. 

    Parameters:
        kx (int or numpy.ndarray): The Fourier mode index in the x direction. Can be a scalar or a numpy array.
        ky (int or numpy.ndarray): The Fourier mode index in the y direction. Can be a scalar or a numpy array.
        a (float): The size of the domain in the x direction. Must be positive.
        b (float): The size of the domain in the y direction. Must be positive.

    Returns:
        numpy.ndarray or float: The wavenumber corresponding to the given mode indices and domain sizes. Returns
        a numpy array if `kx` and `ky` are arrays, otherwise a float.
    """
    if a <= 0 or b <= 0:
        raise ValueError("Domain sizes 'a' and 'b' must be positive and non-zero.")

    # Ensure kx and ky can be treated as arrays for vectorized operations
    kx, ky = np.asarray(kx), np.asarray(ky)

    # Calculate wavenumber
    kz = np.pi * np.sqrt((kx/a)**2 + (ky/b)**2)

    return kz

def solve_for_C1_C2(Fnm, Gnm, kz, c):
    """
    Solve the linear system to find coefficients C1 and C2 for Laplace's equation solution.

    Parameters:
        Fnm (complex): The Fourier coefficient for the function F.
        Gnm (complex): The Fourier coefficient for the function G.
        kz (float): The wavenumber for the current mode.
        c (float): The height of the domain or a characteristic length scale.

    Returns:
        tuple: A tuple containing the coefficients C1 and C2.
    """
    A = np.array([[kz, -kz], [kz * np.exp(kz * c), -kz * np.exp(-kz * c)]])
    b = np.array([Fnm, Gnm])
    C = np.linalg.solve(A, b)

    return C[0], C[1]

def compute_coefficients(box, Fourier_coefficients):
    """
    Compute coefficients for solution to Laplaces equation for all Fourier modes.

    Parameters:
        box (list): The domain of the model specified as [xmin, ymin, zmin, xmax, ymax, zmax].
        Fourier_coefficients (numpy.ndarray): Array containing Fourier modes and their coefficients.

    Returns:
        numpy.ndarray: Array of computed coefficients for the solution to Laplaces equation for each mode.
    """
    # Extract the domain dimensions
    a, b, c = box[3], box[4], box[5]

    # Create empty array to store the results
    Solution_coefficients = []

    for kx, ky, Fnm, Gnm in Fourier_coefficients:
        # Special case handling for zero modes
        if kx == 0 and ky == 0:
            C1, C2 = 0 + 0j, 0 + 0j
        else:
            kz = compute_kz(kx, ky, a, b)
            C1, C2 = solve_for_C1_C2(Fnm, Gnm, kz, c)
        Solution_coefficients.append([kx, ky, C1, C2])

    return Solution_coefficients

def grad_u(x, y, z, coefficients, a, b):
    """
    Calculate the spatial gradient of the solution at a single point (x, y, z).

    This function computes the gradient of the solution for a given point in space,
    based on the Fourier series coefficients. It is optimized for evaluating the
    gradient at individual points rather than across a grid.

    Parameters:
        x (float): The x-coordinate for which to compute the gradient.
        y (float): The y-coordinate for which to compute the gradient.
        z (float): The z-coordinate (height) at which to compute the gradient.
        coefficients (np.ndarray): Array of coefficients for the solution, where
                                   each row contains [kx, ky, C1, C2].
        a (float): Size of the domain in the x direction.
        b (float): Size of the domain in the y direction.

    Returns:
        np.ndarray: The computed gradient of the solution at the given point, as a
                    3-element array corresponding to the gradients [du/dx, du/dy, du/dz].
    """
    # Create an empty array to store the results
    grad = np.zeros(3, dtype=np.complex128)

    for kx, ky, C1, C2 in coefficients:
        kz = compute_kz(kx, ky, a, b)
        term_x = np.exp(1j * np.pi * kx * (x + a) / a)
        term_y = np.exp(1j * np.pi * ky * (y + b) / b)
        exp_kz = np.exp(kz * z)
        exp_neg_kz = np.exp(-kz * z)
        
        grad[0] += (1j * np.pi * kx / a) * term_x * term_y * (C1 * exp_kz + C2 * exp_neg_kz)
        grad[1] += (1j * np.pi * ky / b) * term_x * term_y * (C1 * exp_kz + C2 * exp_neg_kz)
        grad[2] += kz * term_x * term_y * (C1 * exp_kz - C2 * exp_neg_kz)

    return grad

def u(x, y, z, coefficients, a, b):
    """
    Calculate the solution at a single point (x, y, z).

    This function computes the solution for a given point in space,
    based on the Fourier series coefficients. It is optimized for evaluating the
    solution at individual points rather than across a grid.

    Parameters:
        x (float): The x-coordinate for which to compute the gradient.
        y (float): The y-coordinate for which to compute the gradient.
        z (float): The z-coordinate (height) at which to compute the gradient.
        coefficients (np.ndarray): Array of coefficients for the solution, where
                                   each row contains [kx, ky, C1, C2].
        a (float): Size of the domain in the x direction.
        b (float): Size of the domain in the y direction.

    Returns:
        np.ndarray: The computed gradient of the solution at the given point, as a
                    3-element array corresponding to the solution.
    """
    u = 0
    for kx, ky, C1, C2 in coefficients:
        kz = compute_kz(kx, ky, a, b)
        term_x = np.exp(1j * np.pi * kx * (x + a) / a)
        term_y = np.exp(1j * np.pi * ky * (y + b) / b)
        u += term_x * term_y * (C1 * np.exp(kz * z) + C2 * np.exp(-kz * z))
    return u

def map_lattice_points(lattice_points, box, coefficients, A):
    """
    Map lattice points using the gradient of the solution.

    This function updates the positions of a set of lattice points based on the gradient
    of the solution at those points. It is primarily used for mapping points in a
    computational domain according to specific dynamics defined by the gradient.

    Parameters:
        lattice_points (np.ndarray): An array of lattice points with shape (N, 3), 
                                     where N is the number of points, and each point is 
                                     represented as [x, y, z] coordinates.
        coefficients (np.ndarray): Array of coefficients form the solution to Laplaces equation, used to 
                                   calculate the gradient at each point.
        box (list): The domain of the model specified as [xmin, ymin, zmin, xmax, ymax, zmax].
        A (float): Parameter defining the characteristics of the solution, such as amplitude.

    Returns:
        np.ndarray: An array of mapped points with the same shape as the input lattice_points.
    """
    # Extract the domain dimensions
    a, b = box[3], box[4]

    # Initialize an array for the mapped points
    mapped_points = np.zeros_like(lattice_points)

    # Loop over each lattice point
    for i, (x, y, z) in enumerate(lattice_points):
        # Calculate the gradient at the current points
        gradient_basic_u = np.real(grad_u(x, y, z, coefficients, a, b)) + grad_basic_state(x, y, z, A)

        # Map the point forward by the gradient
        mapped_points[i] = [x, y, z] + gradient_basic_u

    return mapped_points