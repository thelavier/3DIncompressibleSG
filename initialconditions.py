import numpy as np
import auxfunctions as aux

# Construct an initial condition by perturbing a steady state
def create_ss_initial(N, B, box, Type):
    """
    Create an initial condition by perturbing a steady state.

    Parameters:
        N (int): The number of seeds.
        B (numpy.ndarray): The steady state transformation matrix.
        box (list): The physical domain of the model [xmin, ymin, zmin, xmax, ymax, zmax].
        Type (str): Type of perturbation to generate:
            - "Thermal Sine": Perturbation with sine waves.
            - "Thermal Gaussian": Perturbation with Gaussian distributions.
            - "None": No perturbation.

    Returns:
        numpy.ndarray: The initial seed positions.

    Raises:
        AssertionError: If an invalid perturbation type is specified.
    """
    # Compute the inverse of B and store for later use
    A = np.linalg.inv(B)

    # Compute the cubic root of the number of seeds to later check that we can generate a valid lattice
    croot = round(N ** (1 / 3))

    # Map the source domain forward under the modified pressure
    # Transform all eight corners of the cube using matrix B
    transformed_corners = []

    # Generate all combinations of minimum and maximum values for each dimension (x, y, z)
    for min_x in [box[0], box[3]]:
        for min_y in [box[1], box[4]]:
            for min_z in [box[2], box[5]]:
                corner = [min_x, min_y, min_z]
                transformed_corner = aux.get_point_transform(corner, B)
                transformed_corners.append(transformed_corner)

    # Convert the list of vectors into a NumPy array
    transformed_corners = np.array(transformed_corners)

    # Use NumPy functions to find the minimum and maximum values for each dimension
    min_values = np.min(transformed_corners, axis=0)
    max_values = np.max(transformed_corners, axis=0)

    # Construct a lattice in the target space
    # Create coordinate arrays for each dimension
    col_0 = np.linspace(min_values[0], max_values[0], croot)
    col_1 = np.linspace(min_values[1], max_values[1], croot)
    col_2 = np.linspace(min_values[2], max_values[2], croot)

    # Create a 3D lattice using meshgrid
    Col_0, Col_1, Col_2 = np.meshgrid(col_0, col_1, col_2)

    # Combine the coordinate arrays into a single matrix
    unperturbed_geostrophic = np.column_stack(
        (Col_0.flatten(), Col_1.flatten(), Col_2.flatten()))

    # Map the lattice back to the fluid domain
    unperturbed_fluid = np.dot(unperturbed_geostrophic, A)

    # Pick the type of perturbation and map back to the geostrophic domain
    if Type == "Thermal Sine":
        x_values = unperturbed_fluid[:, 0]
        y_values = unperturbed_fluid[:, 1]
        perturbed_geostrophic = np.dot(unperturbed_fluid, B) + np.column_stack(
            [np.zeros_like(x_values), np.zeros_like(y_values), np.sin(x_values) + np.sin(y_values)])
    elif Type == "Thermal Gaussian":
        x_values = unperturbed_fluid[:, 0]
        y_values = unperturbed_fluid[:, 1]
        perturbed_geostrophic = np.dot(unperturbed_fluid, B) + np.column_stack(
            [np.zeros_like(x_values), np.zeros_like(y_values), 3 * np.exp(-(x_values ** 2) / 2) + 3 * np.exp(-(y_values ** 2) / 2)])
    elif Type == "None":
        perturbed_geostrophic = np.dot(unperturbed_fluid, B)
    else:
        raise AssertionError("Please specify a valid type of perturbation.")

    # Construct a matrix of perturbations
    perturbation = np.random.uniform(0.8, 1, size=(N, 3))

    return perturbed_geostrophic * perturbation

# Construct Cyclone Initial Condition

def create_cyc_initial(N, box, A, PeriodicX, PeriodicY, PeriodicZ, truncation):
    """
    Create initial conditions for an isolated cyclone with or without shear.

    This function generates a 3D lattice of seed points within the specified domain 'box',
    then computes the perturbation effects and maps the seeds according to the solution's
    gradient, creating initial conditions for cyclone simulation.

    Parameters:
        N (int): The number of seeds.
        box (list): The domain of the model [xmin, ymin, zmin, xmax, ymax, zmax].
        A (float): Shear wind factor, affecting the perturbation (0 or 0.1).
        PeriodicX (bool): If True, apply periodic boundary conditions in the X-direction.
        PeriodicY (bool): If True, apply periodic boundary conditions in the Y-direction.
        PeriodicZ (bool): If True, apply periodic boundary conditions in the Z-direction.
        truncation (int): Determines the number of Fourier series terms to retain.

    Returns:
        numpy.ndarray: An array of the initial seed positions after mapping.

    Raises:
        ValueError: If N is not a perfect cube, which is required for a valid lattice.
    """
    # Calculate FFT coefficients based on cyclone perturbation functions
    fourier_coefficients = aux.compute_fft_coefficients(box, truncation)

    # Solve for C1 and C2 coefficients using the linear system based on Laplace's equation
    solution_coefficients = aux.compute_coefficients(box, fourier_coefficients)

    # Map the lattice points using the gradient of Phi + u
    Zintermediate = aux.map_lattice_points(N, box, solution_coefficients, A)

    # Construct a matrix of perturbations
    perturbation = np.random.uniform(0.99, 1, size=(N, 3))

    # Map the initial condition into the fundamental domain
    Z = aux.get_remapped_seeds(box, Zintermediate * perturbation, PeriodicX, PeriodicY, PeriodicZ)

    return Z