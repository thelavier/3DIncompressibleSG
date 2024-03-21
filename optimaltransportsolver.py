import numpy as np
from pysdot import OptimalTransport
from pysdot.domain_types import ConvexPolyhedraAssembly
from scipy.sparse import csr_matrix

#Constructs a domain to be passed to the laguerre functions
def make_domain(box, PeriodicX, PeriodicY, PeriodicZ):
    """
    Constructs a domain for the optimal transport problem.

    Parameters:
        box (list/tuple): Domain boundaries [xmin, ymin, zmin, xmax, ymax, zmax].
        PeriodicX, PeriodicY, PeriodicZ (bool): Periodicity flags for each axis.

    Returns:
        ConvexPolyhedraAssembly: Domain object for the optimal transport solver.
    """
    domain = ConvexPolyhedraAssembly()
    Lx, Ly, Lz = [box[i+3] - box[i] for i in range(3)]

    # Calculate the offset and size for each dimension based on periodicity
    size = [2 * Lx if PeriodicX else box[3], 
            2 * Ly if PeriodicY else box[4], 
            2 * Lz if PeriodicZ else box[5]]

    offset = [-Lx if PeriodicX else box[0], 
              -Ly if PeriodicY else box[1], 
              -Lz if PeriodicZ else box[2]]

    domain.add_box(offset, size)
    return domain

#Solve the Optimal transport problem and return the centroids and weights
def ot_solve(domain, Y, psi0, err_tol, PeriodicX, PeriodicY, PeriodicZ, box, solver='Petsc', debug=False):
    """
    Solves the optimal transport problem and returns centroids, weights, and cell masses.

    Parameters:
        domain (ConvexPolyhedraAssembly): Source domain of the optimal transport problem.
        Y (numpy.ndarray): Seed positions.
        psi0 (numpy.ndarray): Initial weight guesses.
        err_tol (float): Error tolerance for cell mass.
        PeriodicX, PeriodicY, PeriodicZ (bool): Periodicity flags.
        box (list/tuple): Domain boundaries.
        solver (str): Linear solver to use ('Petsc' or 'Scipy').
        debug (bool): Flag to enable debugging information.

    Returns:
        tuple: Centroids, optimal weights, and cell masses after optimization.
    """
    N = Y.shape[0]
    Lx, Ly, Lz = [abs(box[i+3] - box[i]) for i in range(3)]
    ot = OptimalTransport(positions = Y, weights = psi0, masses = Lx * Ly * Lz * np.ones(N) / N, domain=domain, linear_solver=solver, verbosity=0)
    ot.set_stopping_criterion(err_tol, 'max delta masses')

    # Adding replications based on periodicity
    for x in range(-int(PeriodicX), int(PeriodicX) + 1):
        for y in range(-int(PeriodicY), int(PeriodicY) + 1):
            for z in range(-int(PeriodicZ), int(PeriodicZ) + 1):
                if x != 0 or y != 0 or z != 0:
                    ot.pd.add_replication([Lx * x, Ly * y, Lz * z])

    premass = ot.get_masses() if debug else None
    ot.adjust_weights()
    psi = ot.get_weights()
    postmass = ot.pd.integrals()
    transportcost = ot.pd.second_order_moments()
    mvs = ot.pd.der_centroids_and_integrals_wrt_weight_and_positions()
    m = csr_matrix((mvs.m_values, mvs.m_columns, mvs.m_offsets))

    if debug:
        print('Difference in target and final mass', np.linalg.norm(premass - postmass) / np.linalg.norm(premass))

    return ot.pd.centroids(), psi, postmass, transportcost, m