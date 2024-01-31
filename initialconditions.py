from dolfin import *
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

def create_cyc_initial(N, box, A, PeriodicX, PeriodicY, PeriodicZ):
    """
    Create an initial condition for an isolated cyclone with or without shear.

    Parameters:
        N (int): The number of seeds.
        box (list): The fluid domain of the model [xmin, ymin, zmin, xmax, ymax, zmax].
        A (float): Shear wind factor (0 or 0.1).
        PeriodicX (bool): Periodic boundary condition in the X-direction.
        PeriodicY (bool): Periodic boundary condition in the Y-direction.
        PeriodicZ (bool): Periodic boundary condition in the Z-direction.

    Returns:
        numpy.ndarray: The initial seed positions.

    Raises:
        ValueError: If N does not allow generating a valid lattice.
    """
    # Compute the cubic root of the number of seeds to later check that we can generate a valid lattice
    croot = round(N ** (1 / 3))

    if N == croot ** 3:
        # Create coordinate arrays for each dimension
        col_0 = np.linspace(box[0], box[3], croot)
        col_1 = np.linspace(box[1], box[4], croot)
        col_2 = np.linspace(box[2], box[5], croot)

        # Create a 3D lattice using meshgrid
        Col_0, Col_1, Col_2 = np.meshgrid(col_0, col_1, col_2)

        # Combine the coordinate arrays into a single matrix
        matrix = np.column_stack((Col_0.flatten(), Col_1.flatten(), Col_2.flatten()))

    else:
        raise ValueError('Invalid number of seeds, N must allow generating a valid lattice')

    # Solve Laplace's equation for the perturbation
    a, b, c = box[3], box[4], box[5]

    # Sub domain for Periodic boundary condition
    class PeriodicBoundary(SubDomain):
        """
        Define a periodic boundary condition for a specified rectangular domain.

        This class represents a periodic boundary condition in three dimensions (x, y, z). It allows points on the left (x = -a)
        and bottom (y = -b) sides to be mapped to the right (x = a) and top (y = b) sides of the rectangular domain, respectively.

        Parameters:
            a (float): The domain's half-length in the x-direction.
            b (float): The domain's half-length in the y-direction.

        Methods:
            inside(self, x, on_boundary):
                Determine if a point is inside the periodic boundary.

                Parameters:
                    x (list): The coordinates of the point.
                    on_boundary (bool): Boolean indicating whether the point is on the boundary.

                Returns:
                    bool: True if the point is inside the periodic boundary, False otherwise.

            map(self, x, y):
                Map a point to the opposite side of the periodic boundary.

                Parameters:
                    x (list): The coordinates of the original point.
                    y (list): The coordinates of the mapped point.

                Returns:
                    None
        """
        # Points on the left (x = -a) and bottom (y = -b) are mapped to the right (x = a) and top (y = b)
        def inside(self, x, on_boundary):
            # Use near() to handle floating-point comparisons
            return bool((near(x[0], -a) or near(x[1], -b)) and on_boundary)

        # Map to the opposite side
        def map(self, x, y):
            if near(x[0], a):
                y[0] = x[0] - 2 * a
            else:
                y[0] = x[0]

            if near(x[1], b):
                y[1] = x[1] - 2 * b
            else:
                y[1] = x[1]

            y[2] = x[2]  # z-coordinate remains the same

    # Create mesh and finite elements and enforce periodic boundary conditions
    mesh = BoxMesh(Point(-a, -b, 0), Point(a, b, c), 32, 32, 32)
    V = FunctionSpace(mesh, "CG", 1, constrained_domain=PeriodicBoundary())

    # Sub domain for the Neumann boundary condition
    class Bottom(SubDomain):
        """
        Define a subdomain for the Neumann boundary condition at the bottom of the domain.

        This class represents a subdomain for applying the Neumann boundary condition at the bottom of a 3D domain.
        Points within a small tolerance of z = 0.0 are considered to be on this boundary.

        Methods:
            inside(self, x, on_boundary):
                Determine if a point is inside the bottom subdomain.

                Parameters:
                    x (list): The coordinates of the point.
                    on_boundary (bool): Boolean indicating whether the point is on the boundary.

                Returns:
                    bool: True if the point is inside the bottom subdomain, False otherwise.
        """
        def inside(self, x, on_boundary):
            return bool(near(x[2], 0.0) and on_boundary)

    class Top(SubDomain):
        """
        Define a subdomain for the Neumann boundary condition at the top of the domain.

        This class represents a subdomain for applying the Neumann boundary condition at the top of a 3D domain.
        Points within a small tolerance of z = c are considered to be on this boundary.

        Methods:
            inside(self, x, on_boundary):
                Determine if a point is inside the top subdomain.

                Parameters:
                    x (list): The coordinates of the point.
                    on_boundary (bool): Boolean indicating whether the point is on the boundary.

                Returns:
                    bool: True if the point is inside the top subdomain, False otherwise.
        """
        def inside(self, x, on_boundary):
            return bool(near(x[2], c) and on_boundary)

    # Initialize sub-domain instances
    bottom = Bottom()
    top = Top()

    # Initialize mesh function for boundary domains
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
    boundaries.set_all(0)
    bottom.mark(boundaries, 1)
    top.mark(boundaries, 2)

    # Define expressions for the Neumann conditions
    f_expression = Expression("- 0.6 * ( pow((1 + pow((x[0] + 1) / 0.5, 2) + pow(x[1] / 0.5, 2)), -1.5) - "
                              "0.5 * ( pow(1 + pow(x[0] / 0.5, 2) + pow(x[1] / 0.5, 2), -1.5) + "
                              "pow(1 + pow((x[0] + 2) / 0.5, 2) + pow(x[1] / 0.5, 2), -1.5) ) )", degree=3)  # f(x, y) at top
    g_expression = Expression("0.15 * ( pow(1 + pow(x[0] / 0.5, 2) + pow(x[1] / 0.5, 2), -1.5) - "
                              "0.5 * ( pow(1 + pow((x[0] - 1) / 0.5, 2) + pow(x[1] / 0.5, 2), -1.5) + "
                              "pow(1 + pow((x[0] + 1) / 0.5, 2) + pow(x[1] / 0.5, 2), -1.5) ) )", degree=3)  # g(x, y) at bottom

    # Define expression for Phi(x, y, z)
    phi_expression = Expression("(1 / 2) * (atan2(x[1], 1 + x[2]) - atan2(x[1], 1 - x[2])) - "
                                "0.12 * x[1] * x[2] - (1 / 2) * A * (pow(x[1], 2) - pow(x[2],2))", degree=2, A=A)  # Phi(x, y, z)

    # Define the variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    bform = dot(grad(u), grad(v)) * dx  # bilinear form

    # Set the Neumann Boundary Conditions
    ds = Measure("ds", subdomain_data = boundaries)
    L =  g_expression * v * ds(1) + f_expression * v * ds(2)  # linear form

    # Compute solution
    u = Function(V)
    solve(bform == L, u)

    # Sum the solution of Laplace for the perturbation to the base state
    # Create a non-periodic function space
    V_non_periodic = FunctionSpace(mesh, "CG", 1)
    Phi = Function(V_non_periodic)
    Phi.interpolate(phi_expression)

    # Project or interpolate u onto the non-periodic function space
    u_non_periodic = project(u, V_non_periodic)

    # Add Phi to u in the non-periodic space
    Phi_u = Function(V_non_periodic)
    Phi_u.vector()[:] = Phi.vector()[:] + u_non_periodic.vector()[:]

    # Define the function to compute the mapped lattice points
    def map_lattice_points(lattice_points, function):
        """
        Map lattice points using the gradient of a given function.
    
        This function takes a list of lattice points and maps them forward using the gradient of a provided function.
        It creates a vector function space for the gradient, evaluates the gradient at each lattice point, and maps the
        lattice points forward by the gradient to obtain new mapped points.
    
        Parameters:
            lattice_points (list of lists): A list of lattice points represented as [x, y, z] coordinates.
            function (Expression or Function): The function whose gradient is used for mapping.
    
        Returns:
            numpy.ndarray: An array of mapped points with the same shape as the input lattice_points.
        """
        mapped_points = []

        # Create a vector function space for the gradient
        V_grad = VectorFunctionSpace(mesh, "DG", 1)
        grad_function = project(grad(function), V_grad)

        for point in lattice_points:
            x, y, z = point

            # Create a FEniCS Point object
            fenics_point = Point(float(x), float(y), float(z))

            # Evaluate the gradient of Phi + u at the current lattice point
            gradient_phi_u = grad_function(fenics_point)

            # Map the lattice point forward by the gradient
            mapped_point = [x + gradient_phi_u[0], y + gradient_phi_u[1], z + gradient_phi_u[2]]

            mapped_points.append(mapped_point)

        return np.array(mapped_points)

    # Map the lattice points using the gradient of Phi + u
    Z = aux.get_remapped_seeds(box, map_lattice_points(matrix, Phi_u), PeriodicX, PeriodicY, PeriodicZ)

    return Z