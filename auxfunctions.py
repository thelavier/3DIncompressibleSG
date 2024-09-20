import numpy as np
import msgpack
from scipy.integrate import nquad
import optimaltransportsolver as ots
from pysdot import PowerDiagram
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy import sparse
from scipy.spatial import cKDTree
from petsc4py import PETSc

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

    # Assuming the first entry is a header and should be ignored
    seeds = seeds[1:]
    centroids = centroids[1:]
    weights = weights[1:]
    mass = mass[1:]
    transportcost = transportcost[1:]

    # Find the length of the longest array
    max_length = max(len(arr) for arr in seeds + centroids + weights + mass + transportcost)

    # Check for mismatch and pad shorter arrays with NaNs if necessary
    if any(len(arr) != max_length for arr in seeds + centroids + weights + mass + transportcost):
        print("Mismatch detected. Padding arrays...")
        seeds = [np.pad(arr, ((0, max_length - arr.shape[0]), (0, 0)), 'constant', constant_values=np.nan) for arr in seeds]
        centroids = [np.pad(arr, ((0, max_length - arr.shape[0]), (0, 0)), 'constant', constant_values=np.nan) for arr in centroids]
        weights = [np.pad(arr, (0, max_length - len(arr)), 'constant', constant_values=np.nan) for arr in weights]
        mass = [np.pad(arr, (0, max_length - len(arr)), 'constant', constant_values=np.nan) for arr in mass]
        transportcost = [np.pad(arr, (0, max_length - len(arr)), 'constant', constant_values=np.nan) for arr in transportcost]

    # Convert lists to numpy arrays and exclude the first entry
    Z, C, W, M, TC = np.array(seeds), np.array(centroids), np.array(weights), np.array(mass), np.array(transportcost)

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

def get_properties(Z, C, TC):
    """
    Computes various physical properties based on seed and centroid positions.

    Parameters:
        Z (numpy.ndarray): Seed positions.
        C (numpy.ndarray): Centroid positions.
        TC (numpy.ndarray): Transport cost array.

    Returns:
        tuple: Calculated Meridional Velocities, Zonal Velocities, Total Velocities, Temperature, Kinetic Energy, and Conservation Error.
    """
    # Compute Meridonal Velocities
    MVel = get_velocity(Z, C, 'Meridional')

    # Compute Zonal Velocities
    ZVel = get_velocity(Z, C, 'Zonal')

    # Compute Magnitude of Total Velocity 
    TVel = np.linalg.norm(get_velocity(Z, C, 'Total'), axis = 2)

    # Compute Temperature
    T = Z[:, :, 2]
    
    KineticEnergy = np.sum(TC, axis = 1)
    
    meanEnergy = np.mean(KineticEnergy)

    ConservationError = (meanEnergy - KineticEnergy) / meanEnergy

    return MVel, ZVel, TVel, T, KineticEnergy, ConservationError

def get_centroid_velocity(Z, C, w, box, PeriodicX, PeriodicY, PeriodicZ):
    """
    Calculate the velocity of centroids in a power diagram given positions, weights, 
    and boundary conditions.
    
    This function computes the velocity of centroids in a power diagram, which is 
    derived from the derivatives of centroids and integrals with respect to weights 
    and positions. The computation accounts for periodic boundary conditions along 
    each axis.
    
    Parameters:
    - Z (np.ndarray): The Nx3 array of centroid positions.
    - C (np.ndarray): The Nx3 array of reference positions.
    - w (np.ndarray): The weights associated with each centroid.
    - box (np.ndarray): The bounding box of the domain as a 2x3 array.
    - PeriodicX, PeriodicY, PeriodicZ (bool): Flags indicating periodicity along each axis.
    
    Returns:
    - Cvel (np.ndarray): The calculated velocities of the centroids as a 3N x 1 matrix.
    """
    N = len(Z)

    # Construct power diagram and extract derivatives
    D = ots.make_domain(box, PeriodicX, PeriodicY, PeriodicZ)
    pd = PowerDiagram(positions = Z, weights = w, domain = D)
    mvs = pd.der_centroids_and_integrals_wrt_weight_and_positions()
    m = csr_matrix((mvs.m_values, mvs.m_columns, mvs.m_offsets))

    # Construct DCDw
    row_mask_1 =  np.delete(np.arange(m.shape[0]), np.arange(3, m.shape[0], 4))  # Delete every 4th row
    keep_mask_1 = np.arange(4 * N) % 4 == 3 # Create an keep mask for the columns
    column_mask_1 = np.arange(4*N)[keep_mask_1] # Apply the mask to keep every 4th column
    DCDw = m[row_mask_1][:, column_mask_1]

    # Construct DCDz 
    row_mask_2 = np.delete(np.arange(m.shape[0]), np.arange(3, m.shape[0], 4)) # Delete every 4th row
    exclude_mask_2 = (np.arange(4 * N) - 3) % 4 == 0 # Create an exclude mask for the columns
    column_mask_2 = np.arange(4*N)[~exclude_mask_2] # Invert the mask to delete every 4th column
    DCDz = m[row_mask_2][:, column_mask_2]

    # Construct DmDw
    row_mask_3 = range(3, m.shape[0], 4) # Keep every 4th row
    keep_mask_3 = np.arange(4 * N) % 4 == 3 # Create an keep mask for the columns
    column_mask_3 = np.arange(4 * N)[keep_mask_3] # Apply the mask to keep every 4th column
    DmDw = m[row_mask_3][:, column_mask_3]

    # Construct DmDz
    row_mask_4 = range(3, m.shape[0], 4) # Keep every 4th row
    exclude_mask_4 = (np.arange(4 * N) - 3) % 4 == 0 # Create an exclude mask for the columns
    column_mask_4 = np.arange(4*N)[~exclude_mask_4] # Invert the mask to delete every 4th column
    DmDz = m[row_mask_4][:, column_mask_4]

    # Construct DzDt
    P = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
    J = sparse.kron(sparse.eye(N, dtype=int), sparse.csr_matrix(P))
    DzDt = J.dot((Z - C).flatten())

    # Compute the Centroid Velocities
    Cvel = (DCDw.dot(-spsolve(DmDw, DmDz)) + DCDz).dot(DzDt)

    return Cvel

def add_points_ensuring_distance(M_points, N):
    M = len(M_points)
    target_distance = 1 / np.sqrt(N + M)
    new_points = []
    tree = cKDTree(M_points)  # Create a k-d tree from existing points
    
    attempts = 0
    max_attempts = 100000  # Increased attempts for more chances

    while len(new_points) < N and attempts < max_attempts:
        new_point = np.random.uniform(-1, 1, (1, 3))
        # Use the k-d tree to find the distance to the nearest point
        closest_dist, _ = tree.query(new_point)
        if closest_dist > target_distance:
            new_points.append(new_point[0])
            # Update the tree with the new point
            tree = cKDTree(np.vstack([M_points, new_points]))
        attempts += 1

    if len(new_points) < N:
        print("Failed to place all new points within the maximum attempts allowed.")
    else:
        print("All new points placed successfully.")

    all_points = np.vstack((M_points, new_points))
    return all_points

def get_comparison_indices(Ndt, NdtRef, tf):
    """
    Computes the indices for comparison at specific intervals from two time series.

    Parameters:
    - Ndt (int): The number of data points in the time series being inspected.
    - NdtRef (int): The number of data points in the reference time series.
    - tf (float): The final time up to which the data points are recorded.

    Returns:
    - list: A list of tuples where each tuple contains two integers representing the indices in the inspected and reference 
             time series respectively, corresponding to the comparison times.
    """
    indices = []
    for step in range(Ndt):
        comptime = (step / Ndt) * tf
        ind = step
        indRef = min(int(round((NdtRef / tf) * comptime)), NdtRef - 1)
        indices.append((ind, indRef))
    return indices

def solve(A, b):
    """
    Solve a linear system Ax = b using PETSc's Conjugate Gradient (CG) method
    with Geometric Algebraic MultiGrid (GAMG) preconditioning.

    This function is designed to solve sparse linear systems that arise from
    discretizations of PDEs, which are common in scientific computing. CG is
    an iterative method suitable for systems where A is symmetric positive-definite,
    and GAMG is used to accelerate convergence.

    Parameters:
    - A (PETSc.Mat): The matrix representing the linear system.
    - b (PETSc.Vec): The right-hand side vector in the linear system.

    Returns:
    - x (PETSc.Vec): The solution vector that satisfies Ax = b.

    Note:
    This function assumes that the matrix A is square, symmetric, and positive-definite.
    The function will return a solution vector using the CG method, which may not
    converge if these conditions are not met. It is up to the caller to ensure that
    the matrix A fits the method's requirements.
    """
    # Create a sequential vector of the given size to store the solution of the linear system
    size = A.getSizes()[0]
    x = PETSc.Vec().createSeq(size)

    # Create a KSP (Krylov subspace solver) object, which provides an interface to the Krylov subspace methods for solving linear systems
    ksp = PETSc.KSP().create()
    ksp.setType('cg') # Set the type of the KSP solver to Conjugate Gradient (CG)
    ksp.getPC().setType('gamg') # Get the preconditioner context and set its type to Geometric Algebraic MultiGrid (GAMG)

    # Set the operators (matrix and preconditioner) and options for the KSP solver
    ksp.setOperators(A)
    ksp.setTolerances(rtol=1e-5)
    ksp.setFromOptions()

    # Solve the linear system Ax = b, and store
    ksp.solve(b, x)

    return x

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
    Compute the gradient of the basic state function at given point(s).

    This function calculates the spatial gradient of the basic state for scalar or array inputs.
    It is instrumental in seeding initial conditions within the geostrophic flow, allowing for
    analysis of how small perturbations might develop over time in both individual points and
    spatial grids.

    Parameters:
        x (float or np.ndarray): The x-coordinate(s) of the point(s), where the gradient with respect
                                 to x is always 0, indicating no variation in the x-direction.
        y (float or np.ndarray): The y-coordinate(s) of the point(s), influencing the gradient in the y-direction.
        z (float or np.ndarray): The z-coordinate(s) (height or depth) of the point(s), affecting the gradient in the z-direction.
        A (float): Shear wind factor, a parameter that influences the computation of the gradient
                   and reflects environmental conditions.

    Returns:
        np.ndarray: A 3-element array (or an array of 3-element arrays for multiple points) representing
                    the gradient of the basic state at the given point(s), with elements corresponding
                    to [du/dx, du/dy, du/dz]. The first element (du/dx) is always 0.

    Note:
        This function supports both scalar and array inputs, enhancing its applicability for a wide
        range of scenarios, from evaluating individual points to processing spatial grids.
    """
    # Determine the shape based on input types, handling both scalar and array inputs
    if np.isscalar(x) and np.isscalar(y) and np.isscalar(z):
        grad = np.zeros(3, dtype=np.float64)  # For scalar inputs
    else:
        # Ensure compatibility with arrays of different shapes
        shape = np.broadcast(np.asarray(x), np.asarray(y), np.asarray(z)).shape
        grad = np.zeros(shape + (3,), dtype=np.float64)  # For array inputs

    # Compute the gradient
    grad[..., 0] = 0 #dPhi/dx
    grad[..., 1] = -A * y - 0.12 * z - 0.5 * (1 - z) / (y**2 + (1 - z)**2) + 0.5 * (z + 1) / (y**2 + (z + 1)**2) #dPhi/dy
    grad[..., 2] = A * z - 0.12 * y - 0.5 * y / (y**2 + (z + 1)**2) - 0.5 * y / (y**2 + (1 - z)**2) #dPhi/dz

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
    Calculate the spatial gradient of the solution at a single point or for arrays of points (x, y, z).

    This function computes the gradient of the solution for a given point(s) in space,
    based on the Fourier series coefficients. It supports both scalar and array inputs
    for x, y, and z coordinates.

    Parameters:
        x (float or np.ndarray): The x-coordinate(s) for which to compute the gradient.
        y (float or np.ndarray): The y-coordinate(s) for which to compute the gradient.
        z (float or np.ndarray): The z-coordinate(s) (height) at which to compute the gradient.
        coefficients (np.ndarray): Array of coefficients for the solution, where
                                   each row contains [kx, ky, C1, C2].
        a (float): Size of the domain in the x direction.
        b (float): Size of the domain in the y direction.

    Returns:
        np.ndarray: The computed gradient of the solution at the given point(s), as an
                    array corresponding to the gradients [du/dx, du/dy, du/dz].
    """
    # Determine if the inputs are scalar or array to initialize the gradient storage appropriately
    if np.isscalar(x) and np.isscalar(y) and np.isscalar(z):
        grad = np.zeros(3, dtype=np.complex128)  # For scalar inputs
    else:
        # Ensures compatibility with arrays of different shapes
        shape = np.broadcast(np.asarray(x), np.asarray(y), np.asarray(z)).shape
        grad = np.zeros(shape + (3,), dtype=np.complex128)  # For array inputs

    # Compute the gradient
    for kx, ky, C1, C2 in coefficients:
        kz = compute_kz(kx, ky, a, b)
        term_x = np.exp(1j * np.pi * kx * (x + a) / a)
        term_y = np.exp(1j * np.pi * ky * (y + b) / b)
        grad[..., 0] += (1j * np.pi * kx / a) * term_x * term_y * (C1 * np.exp(kz * z) + C2 * np.exp(-kz * z))  # du/dx
        grad[..., 1] += (1j * np.pi * ky / b) * term_x * term_y * (C1 * np.exp(kz * z) + C2 * np.exp(-kz * z))  # du/dy
        grad[..., 2] += term_x * term_y * kz * (C1 * np.exp(kz * z) - C2 * np.exp(-kz * z))  # du/dz

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

def map_lattice_points(N, box, coefficients, A):
    """
    Map lattice points using the gradient of the solution.

    This function updates the positions of a set of lattice points based on the gradient
    of the solution at those points. It is primarily used for mapping points in a
    computational domain according to specific dynamics defined by the gradient.

    Parameters:
        N (int): Number of points
        coefficients (np.ndarray): Array of coefficients form the solution to Laplaces equation, used to 
                                   calculate the gradient at each point.
        box (list): The domain of the model specified as [xmin, ymin, zmin, xmax, ymax, zmax].
        A (float): Parameter defining the characteristics of the solution, such as amplitude.

    Returns:
        np.ndarray: An array of mapped points with the same shape as the input lattice_points.
    """
    # Compute the cubic root of the number of seeds to later check that we can generate a valid lattice
    croot = round(N ** (1 / 3))

    if N == croot ** 3:
        # Create coordinate arrays for each dimension
        col_x = np.linspace(box[0], box[3], croot)
        col_y = np.linspace(box[1], box[4], croot)
        col_z = np.linspace(box[2], box[5], croot)

        # Create a 3D lattice using meshgrid
        # Pay attention to the indexing scheme. The lattice points and the gradient of the basic state are computed
        # using the 'ij' indexing but the gradient of u is computed using the 'xy' indexing. This means that in order 
        # to correctly combine the there matrices the gradient of u needs to be transposed with indexing 'ij' and 
        # the lattice points and gradient of phi need to be transposed with indexing 'xy'.
        Col_X1, Col_Y1, Col_Z1 = np.meshgrid(col_x, col_y, col_z, indexing='ij') 

        # Combine the coordinate arrays into a single matrix
        lattice_points = np.stack((Col_X1, Col_Y1, Col_Z1), axis=-1)

    else:
        raise ValueError('Invalid number of seeds, N must allow generating a valid lattice')

    # Extract the domain dimensions
    a, b = box[3], box[4]

    mapped_points = np.zeros((croot, croot, croot) + (3,), dtype=np.float64)  # For array inputs

    # Compute the mapped forward lattice points
    gradu_values = np.real(grad_u(col_x[:, None, None], col_y[None, :, None], col_z[None, None, :], coefficients, a, b)).T
    gradphi_values = grad_basic_state(col_x[:, None, None], col_y[None, :, None], col_z[None, None, :], A)
    mapped_points[:, :, :, 0] = lattice_points[:, :, :, 0] + gradphi_values[:, :, :, 0] + gradu_values[0, :, :, :]  
    mapped_points[:, :, :, 1] = lattice_points[:, :, :, 1] + gradphi_values[:, :, :, 1] + gradu_values[1, :, :, :]  
    mapped_points[:, :, :, 2] = lattice_points[:, :, :, 2] + gradphi_values[:, :, :, 2] + gradu_values[2, :, :, :]  

    return mapped_points.reshape(-1, 3)

def check_sparsity_petsc(A):
    # Get the sizes and number of nonzeros
    m, n = A.getSize()
    info = A.getInfo()  # Get information dictionary about the matrix
    nnz = info['nz_used']  # Number of nonzeros actually used
    total_elements = m * n
    sparsity = nnz / total_elements
    print(f"Matrix Sparsity: {sparsity:.4f} (nonzero elements/total elements)")

def check_symmetry_petsc(A):
    # Using MatTranspose to get the transpose of A
    AT = A.transpose()  # This method handles allocation and returns the transposed matrix

    # Create a matrix to store A - AT
    M_diff = A.copy()
    M_diff.axpy(-1, AT)  # M_diff = A - AT

    # Calculate the Frobenius norm of the difference
    norm = M_diff.norm(PETSc.NormType.FROBENIUS)
    is_symmetric = norm == 0
    print(f"Matrix is Symmetric: {is_symmetric}")

def check_definiteness_petsc(A):
    ksp = PETSc.KSP().create()
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.CHOLESKY)
    try:
        pc.setOperators(A)
        pc.setUp()
        is_definite = True
    except Exception as e:
        print(f"Failed to perform Cholesky factorization: {str(e)}")
        is_definite = False
    print(f"Matrix is Positive Definite: {is_definite}")