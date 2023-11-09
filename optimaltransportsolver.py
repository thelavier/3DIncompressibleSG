import numpy as np
from pysdot import OptimalTransport
from pysdot.domain_types import ConvexPolyhedraAssembly

#Constructs a domain to be passed to the laguerre functions
def make_domain(box, PeriodicX, PeriodicY, PeriodicZ):
    """
    Function returning the source domain for the optimal tranpsort problem.

    Inputs:
        box: list or tuple defining domain [xmin, ymin, zmin, xmax, ymax, zmax]
        PeriodicX: a boolian indicating if the boundaries are periodic in x 
        PeriodicY: a boolian indicating if the boundaries are periodic in y
        PeriodicZ: a boolian indicating if the boundaries are periodic in z

    Outputs:
        domain: domain object for passing to optimal transport solver
    """
    domain = ConvexPolyhedraAssembly()
    
    Lx = box[3] - box[0]
    Ly = box[4] - box[1]
    Lz = box[5] - box[2]

    if PeriodicX == False and PeriodicY == False and PeriodicZ == False:
        domain.add_box([box[0], box[1], box[2]], [box[3], box[4], box[5]])

    elif PeriodicX == True and PeriodicY == False and PeriodicZ == False: 
        domain.add_box([-Lx, box[1], box[2]], [2*Lx, box[4], box[5]])

    elif PeriodicX == False and PeriodicY == True and PeriodicZ == False: 
        domain.add_box([box[0], -Ly, box[2]], [box[3], 2*Ly, box[5]])

    elif PeriodicX == False and PeriodicY == False and PeriodicZ == True: 
        domain.add_box([box[0], box[1], -Lz], [box[3], box[4], 2*Lz])

    elif PeriodicX == False and PeriodicY == True and PeriodicZ == True: 
        domain.add_box([box[0], -Ly, -Lz], [box[3], 2*Ly, 2*Lz])

    elif PeriodicX == True and PeriodicY == False and PeriodicZ == True: 
        domain.add_box([-Lx, box[1], -Lz], [2*Lx, box[4], 2*Ly])

    elif PeriodicX == True and PeriodicY == True and PeriodicZ == False: 
        domain.add_box([-Lx, -Ly, box[2]], [2*Lx, 2*Ly, box[5]])

    elif PeriodicX == True and PeriodicY == True and PeriodicZ == True: 
        domain.add_box([-Lx, -Ly, -Lz], [2*Lx, 2*Ly, 2*Lz])

    else: 
        AssertionError('Please specify the periodicity of the domain.')

    return domain

#Solve the Optimal transport problem and return the centroids and weights
def ot_solve(domain, Y, psi0, err_tol, PeriodicX, PeriodicY, PeriodicZ, box, solver = 'Petsc', debug = False):
    """
    Function solving the optimal transport problem using the Damped Newton Method and returning the centroids of the optimal diagram.

    Inputs:
        domain: The source domain of the optimal transport problem
        mass: the target volume
        Y: The seed positions 
        psi0: The inital weight guess for each seed
        err_tol: The error tolerance on the mass of the cells
        PeriodicX: a boolian indicating if the boundaries are periodic in x 
        PeriodicY: a boolian indicating if the boundaries are periodic in y
        PeriodicZ: a boolian indicating if the boundaries are periodic in z
        box: list or tuple defining domain [xmin, ymin, zmin, xmax, ymax, zmax] 
        solver: a string indicating whether we are using the Petsc linear solver or the Scipy linear solver
        debug: a boolian indicating if the code is running in debug mode

    Outputs:
        centroids: The centroids of the optimal Laguerre diagram
        psi: The optimal weights
    """
    N = Y.shape[0] #Determine the number of seeds
    Lx = box[3] - box[0]
    Ly = box[4] - box[1]
    Lz = box[5] - box[2]

    if PeriodicX == False and PeriodicY == False and PeriodicZ == False:
        ot = OptimalTransport(positions = Y, weights = psi0, masses = domain.measure() / N * np.ones(N), domain = domain, linear_solver = solver) #Establish the Optimal Transport problem
        ot.set_stopping_criterion(err_tol, 'max delta masses') #Pick the stopping criterion to be the mass of the cells

    elif PeriodicX == True and PeriodicY == True and PeriodicZ == False:
        ot = OptimalTransport(positions = Y, weights = psi0, masses = Lx * Ly * Lz * np.ones(N) / N, domain = domain, linear_solver = solver) #Establish the Optimal Transport problem
        ot.set_stopping_criterion(err_tol, 'max delta masses') #Pick the stopping criterion to be the mass of the cells
        for x in [ -1, 0, 1 ]:
            for y in [ -1, 0, 1 ]:
                if x or y:
                    ot.pd.add_replication( [ Lx * x, Ly * y, 0 ] )

    elif PeriodicX == True and PeriodicY == False and PeriodicZ == True:
        ot = OptimalTransport(positions = Y, weights = psi0, masses = Lx * Ly * Lz * np.ones(N) / N, domain = domain, linear_solver = solver) #Establish the Optimal Transport problem
        ot.set_stopping_criterion(err_tol, 'max delta masses') #Pick the stopping criterion to be the mass of the cells
        for x in [ -1, 0, 1 ]:
            for z in [ -1, 0, 1 ]:
                if x or z:
                    ot.pd.add_replication( [ Lx * x, 0, Lz * z ] )

    elif PeriodicX == False and PeriodicY == True and PeriodicZ == True:
        ot = OptimalTransport(positions = Y, weights = psi0, masses = Lx * Ly * Lz * np.ones(N) / N, domain = domain, linear_solver = solver) #Establish the Optimal Transport problem
        ot.set_stopping_criterion(err_tol, 'max delta masses') #Pick the stopping criterion to be the mass of the cells
        for y in [ -1, 0, 1 ]:
            for z in [ -1, 0, 1 ]:
                if y or z:
                    ot.pd.add_replication( [ 0, Ly * y, Lz * z ] )

    elif PeriodicX == False and PeriodicY == False and PeriodicZ == True:
        ot = OptimalTransport(positions = Y, weights = psi0, masses = Lx * Ly * Lz * np.ones(N) / N, domain = domain, linear_solver = solver) #Establish the Optimal Transport problem
        ot.set_stopping_criterion(err_tol, 'max delta masses') #Pick the stopping criterion to be the mass of the cells
        for z in [ -1, 1 ]:
            ot.pd.add_replication( [ 0, 0, Lz * z ] )

    elif PeriodicX == False and PeriodicY == True and PeriodicZ == False:
        ot = OptimalTransport(positions = Y, weights = psi0, masses = Lx * Ly * Lz * np.ones(N) / N, domain = domain, linear_solver = solver) #Establish the Optimal Transport problem
        ot.set_stopping_criterion(err_tol, 'max delta masses') #Pick the stopping criterion to be the mass of the cells
        for y in [ -1, 1 ]:
            ot.pd.add_replication( [ 0, Ly * y, 0 ] )

    elif PeriodicX == True and PeriodicY == False and PeriodicZ == False:
        ot = OptimalTransport(positions = Y, weights = psi0, masses = Lx * Ly * Lz * np.ones(N) / N, domain = domain, linear_solver = solver) #Establish the Optimal Transport problem
        ot.set_stopping_criterion(err_tol, 'max delta masses') #Pick the stopping criterion to be the mass of the cells
        for x in [ -1, 1 ]:
            ot.pd.add_replication( [ Lx * x, 0, 0 ] )

    elif PeriodicX == True and PeriodicY == True and PeriodicZ == True:
        ot = OptimalTransport(positions = Y, weights = psi0, masses = Lx * Ly * Lz * np.ones(N) / N, domain = domain, linear_solver = solver) #Establish the Optimal Transport problem
        ot.set_stopping_criterion(err_tol, 'max delta masses') #Pick the stopping criterion to be the mass of the cells
        for x in [ -1, 0, 1 ]:
            for y in [ -1, 0, 1 ]:
                for z in [-1, 0, 1]:
                    if x or y or z:
                        ot.pd.add_replication( [ Lx * x, Ly * y, Lz * z ] )
    
    else:
        AssertionError('Please specify the periodicity')

    if debug == True:
        premass = ot.get_masses() # Extract the target masses
    #    print('Target masses before Damped Newton', premass)
    #    print('Weights before Damped Newton', ot.get_weights())
    #    print('Mass before Damped Newton', ot.pd.integrals())
        print('Total:', sum(ot.pd.integrals()))
    else:
        pass

    ot.adjust_weights() # Use Damped Newton to find the optimal weight
    psi = ot.get_weights() # Extract the optimal weights from the solver
    postmass = ot.pd.integrals() # Extract the mass of the cells after Damped Newton

    if debug == True:
    #    print('Mass after Damped Newton', postmass, 'Total:', sum(postmass)) #Print the mass of each cell
        print('Difference in target and final mass', np.linalg.norm(premass-postmass)) #Check how different the final mass is from the target mass
    else:
        pass

    return ot.pd.centroids(), psi, postmass

