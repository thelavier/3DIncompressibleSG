import numpy as np

#Construct initial condition
def create_initial(N, maxx, maxy, maxz, minx, miny, minz, Type):
    """
    Function that constructs an initial condition. Allows for different distributions on different axes.

    Inputs:
        N: The number of seeds
        maxx: The maximum position in the x direction
        maxy: The maximum position in the y direction
        maxz: The maximum position in the z direction
        minx: The minimum position in the x direction
        miny: The minimum position in the y direction
        minz: The minimum position in the z direction
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
        perturbation = np.random.uniform(0.8, 1, size = (N, 3))

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
        perturbation = np.random.uniform(0.8, 1, size = (N, 3))

        return matrix * perturbation

    else:
        raise ValueError('Please specify the type of initial condition you want to use and make sure the number of seeds can generate a valid lattice.')
