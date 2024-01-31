from dolfin import *
import numpy as np
import auxfunctions as aux

# Construct an artificial initial condition

def create_ss_initial(N, B, box, Type):
    """
    Function that constructs an initial condition. Allows for different distributions on different axes.

    Inputs:
        N: The number of seeds
        B: The steady state (a matrix)
        box: The physical domain of the model
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
    unperturbed_geostrophic = np.column_stack(
        (Col_0.flatten(), Col_1.flatten(), Col_2.flatten()))

    # Map the latice back to the fluid domain
    unperturbed_fluid = np.dot(unperturbed_geostrophic, A)

    # Pick the type of perturbation and map back to the geostrophic domain
    match Type:
        case "Thermal Sine":
            x_values = unperturbed_fluid[:, 0]
            y_values = unperturbed_fluid[:, 1]
            perturbed_geostrophic = np.dot(unperturbed_fluid, B) + np.column_stack(
                [np.zeros_like(x_values), np.zeros_like(y_values), np.sin(x_values) + np.sin(y_values)])
        case "Thermal Gaussian":
            x_values = unperturbed_fluid[:, 0]
            y_values = unperturbed_fluid[:, 1]
            perturbed_geostrophic = np.dot(unperturbed_fluid, B) + np.column_stack([np.zeros_like(x_values), np.zeros_like(y_values), 3 * np.exp(-(x_values ** 2) / 2) + 3 * np.exp(-(y_values ** 2) / 2)])
        case "None":
            perturbed_geostrophic = np.dot(unperturbed_fluid, B)
        case _:
            AssertionError("Please specify a valid type of perturbation.")
    
    # Construct matrix of perturbations
    perturbation = np.random.uniform(0.8, 1, size=(N, 3))

    return perturbed_geostrophic * perturbation

# Construct Cyclone Initial Condition

def create_cyc_initial(N, box, A):
    """
    Function that constructs the initial condition for an isolated cyclone with or without shear.

    Inputs:
        N: The number of seeds
        box: The fluid domain of the model given as [xmin, ymin, zmin, xmax, ymax, zmax]
        A: Either 0 or 0.1 indicating if there is a shear wind
        pert: A number between 2 and 0 indicating the strength of the perturbation, where 1 is no perturbation

    Outputs:
        matrix: The initial seeds positions
    """
    # Compute the square root of the number of seeds to later check that we can generate a valid lattice
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
        raise ValueError(
            'Please provide a number of columns that generates a valid lattice')

    # Solve Laplace's equation for the perturbation
    a, b, c = box[3], box[4], box[5]

    # Sub domain for Periodic boundary condition
    class PeriodicBoundary(SubDomain):
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
        def inside(self, x, on_boundary):
            return bool(near(x[2], 0.0) and on_boundary)

    class Top(SubDomain):
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
    Z = map_lattice_points(matrix, Phi_u)

    return Z