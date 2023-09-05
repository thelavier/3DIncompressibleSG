import numpy as np
import auxfunctions as aux

#Construct an artificial initial condition
def create_artificial_initial(N, minx, miny, minz, maxx, maxy, maxz, Type):
    """
    Function that constructs an initial condition. Allows for different distributions on different axes.

    Inputs:
        N: The number of seeds
        minx: The minimum position in the x direction
        miny: The minimum position in the y direction
        minz: The minimum position in the z direction
        maxx: The maximum position in the x direction
        maxy: The maximum position in the y direction
        maxz: The maximum position in the z direction
        Type: Type of initial condition to generate

    Outputs:
        matrix: The initial seeds positions
    """

    #Compute the cubic root of the number of seeds to later check that we can generate a valid lattice
    croot = round(N ** (1 / 3))

    if Type == 'uniform wsp':
        # Generate random values for the first and second columns
        col_0 = np.random.uniform(minx, maxx, size=N)
        col_1 = np.random.uniform(miny, maxy, size=N)

        # Generate random values for the third column
        col_2 = np.random.uniform( 2 * np.sin(col_0), 2 * np.sin(col_1), size=N)

        # Create the matrix by concatenating the columns
        matrix = np.column_stack((col_0, col_1, col_2))

        return matrix

    elif Type == 'uniform':
        # Generate random values for the first and second columns
        col_0 = np.random.uniform(minx, maxx, size=N)
        col_1 = np.random.uniform(miny, maxy, size=N)

        # Generate random values for the third column
        col_2 = np.random.uniform(minz, maxz, size=N)

        # Create the matrix by concatenating the columns
        matrix = np.column_stack((col_0, col_1, col_2))

        return matrix

    elif Type == 'normal':
        # Generate random values for the first and second columns
        col_0 = np.random.normal(0, maxx, size=N)
        col_1 = np.random.normal(0, maxy, size=N)

        # Generate random values for the third column
        col_2 = np.random.normal(0, maxz, size=N)

        # Create the matrix by concatenating the columns
        matrix = np.column_stack((col_0, col_1, col_2))

        return matrix

    elif Type == 'linear':
            # Generate  values for the first and second columns
            col_0 = np.linspace(minx, maxx, N)
            col_1 = np.linspace(miny, maxy, N)
    
            # Generate random values for the third column
            col_2 = np.linspace(minz, maxz, N)
    
            # Create the matrix by concatenating the columns
            matrix = np.column_stack((col_0, col_1, col_2))
    
            return matrix

    elif Type == 'linear wsp':
        # Generate  values for the first and second columns
        col_0 = np.linspace(minx, maxx, N)
        col_1 = np.linspace(miny, maxy, N)

        # Generate random values for the third column
        col_2 = 2 * np.sin(np.linspace(minz, maxz, N))

        # Create the matrix by concatenating the columns
        matrix = np.column_stack((col_0, col_1, col_2))

        return matrix

    elif Type == 'lattice' and N == croot ** 3:
        # Create coordinate arrays for each dimension
        col_0 = np.linspace(minx, maxx, croot)
        col_1 = np.linspace(miny, maxy, croot)
        col_2 = np.linspace(minz, maxz, croot)

        # Create a 3D lattice using meshgrid
        Col_0, Col_1, Col_2 = np.meshgrid(col_0, col_1, col_2)

        # Combine the coordinate arrays into a single matrix
        matrix = np.column_stack((Col_0.flatten(), Col_1.flatten(), Col_2.flatten()))

        # Construct matrix of perturbations
        perturbation = np.random.uniform(0.9, 1, size = (N, 3))

        return matrix * perturbation

    elif Type == 'lattice wsp' and N == croot ** 3:
        # Create coordinate arrays for each dimension
        col_0 = np.linspace(minx, maxx, croot)
        col_1 = np.linspace(miny, maxy, croot)
        col_2 = np.linspace(minz, maxz, croot)

        # Create a 3D lattice using meshgrid
        Col_0, Col_1, Col_2 = np.meshgrid(col_0, col_1, col_2)

        # Transform the Z corrdinates to make a sine perturbation
        Col_2 = 2 * np.sin(Col_0) * np.sin(Col_1)

        # Combine the coordinate arrays into a single matrix
        matrix = np.column_stack((Col_0.flatten(), Col_1.flatten(), Col_2.flatten()))

        # Construct matrix of perturbations
        perturbation = np.random.uniform(0.9, 1, size = (N, 3))

        return matrix * perturbation

    else:
        raise ValueError('Please specify the type of initial condition you want to use and make sure the number of seeds can generate a valid lattice.')

#Construct an initial condition that is a perturbation of a steady state
def create_ss_initial(N, B, box, Type):
    """
    Function that constructs an initial condition. Allows for different distributions on different axes.

    Inputs:
        N: The number of seeds
        B: The steady state (a matrix)
        box: 
        Type: Type of perturbation to generate

    Outputs:
        matrix: The initial seeds positions
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
    unperturbed_geostrophic = np.column_stack((Col_0.flatten(), Col_1.flatten(), Col_2.flatten()))

    # Map the latice back to the fluid domain
    unperturbed_fluid = np.dot(unperturbed_geostrophic, A)

    # Pick the type of perturbation and map back to the geostrophic domain
    match Type:
        case "Thermal Sine":
            x_values = unperturbed_fluid[:, 0]
            y_values = unperturbed_fluid[:, 1]
            perturbed_geostrophic = np.dot(unperturbed_fluid, B) + np.column_stack([np.zeros_like(x_values), np.zeros_like(y_values), np.sin(x_values) + np.sin(y_values)])
        case "Thermal Gaussian":
            x_values = unperturbed_fluid[:, 0]
            y_values = unperturbed_fluid[:, 1]
            perturbed_geostrophic = np.dot(unperturbed_fluid, B) + np.column_stack([np.zeros_like(x_values), np.zeros_like(y_values), 3 * np.exp(-(x_values ** 2) / 2) + 3 * np.exp(-(y_values ** 2) / 2)])
        case "None":
            perturbed_geostrophic = np.dot(unperturbed_fluid, B)
        case _:
            AssertionError("Please specify a valid type of perturbation.")

    return perturbed_geostrophic