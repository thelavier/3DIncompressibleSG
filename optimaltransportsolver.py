import numpy as np
import pysdot
from pysdot import OptimalTransport
from pysdot.domain_types import ConvexPolyhedraAssembly

#Constructs a domain to be passed to the laguerre functions
def make_domain(box=[-2, -2, 0, 2, 2, 1]):
    """
    Function returning the source domain for the optimal tranpsort problem.

    Inputs:
        box: list or tuple defining domain [xmin, ymin, zmin, xmax, ymax, zmax]
        img: the measure

    Outputs:
        domain: domain object for passing to optimal transport solver
    """
    domain = ConvexPolyhedraAssembly()
    domain.add_box([box[0], box[1], box[2]], [box[3], box[4], box[5]])
    return domain

#Find the centroids of a Laguerre Diagram
def laguerre_centroids(domain, Y, psi):
    """
    Function returning the centroids of a Laguerre diagram.

    Inputs:
        domain: The source domain of the optimal transport problem
        Y: The seed positions 
        psi: The corresponding weights for each seed

    Outputs:
        centroids: the centroids of the Laguerre diagram
    """
    return pysdot.PowerDiagram(Y, psi, domain).centroids()

#Solve the Optimal transport problem and return the centroids and weights
def ot_solve(domain, Y, psi0, err_tol, PeriodicX, PeriodicY, PeriodicZ):
    """
    Function solving the optimal transport problem using the Damped Newton Method and returning the centroids of the optimal diagram.

    Inputs:
        domain: The source domain of the optimal transport problem
        Y: The seed positions 
        psi0: The inital weight guess for each seed
        err_tol: The error tolerance on the mass of the cells
        PeriodicX: a boolian indicating if the boundaries are periodic in x 
        PeriodicY: a boolian indicating if the boundaries are periodic in y
        PeriodicZ: a boolian indicating if the boundaries are periodic in z

    Outputs:
        centroids: The centroids of the optimal Laguerre diagram
        psi: The optimal weights
    """
    N = Y.shape[0] #Determine the number of seeds
    ot = OptimalTransport(positions = Y, weights = psi0, masses = domain.measure() / N * np.ones(N), domain = domain, linear_solver= 'Petsc') #Establish the Optimal Transport problem
    #ot = OptimalTransport(positions = Y, weights = psi0, masses = np.ones(N) / N, domain = domain, linear_solver= 'Petsc') #Establish the Optimal Transport problem
    ot.set_stopping_criterion(err_tol, 'max delta masses') #Pick the stopping criterion to be the mass of the cells

    if PeriodicX == False and PeriodicY == False and PeriodicZ == False:
        pass

    elif PeriodicX == True and PeriodicY == True and PeriodicZ == False:
        for x in [ -1, 0, 1 ]:
            for y in [ -1, 0, 1 ]:
                if x or y:
                    ot.pd.add_replication( [ x, y, 0 ] )

    elif PeriodicX == True and PeriodicY == False and PeriodicZ == True:
        for x in [ -1, 0, 1 ]:
            for z in [ -1, 0, 1 ]:
                if x or z:
                    ot.pd.add_replication( [ x, 0, z ] )

    elif PeriodicX == False and PeriodicY == True and PeriodicZ == True:
        for y in [ -1, 0, 1 ]:
            for z in [ -1, 0, 1 ]:
                if y or z:
                    ot.pd.add_replication( [ 0, y, z ] )

    elif PeriodicX == False and PeriodicY == False and PeriodicZ == True:
        for z in [ -1, 1 ]:
            ot.pd.add_replication( [ 0, 0, z ] )

    elif PeriodicX == False and PeriodicY == True and PeriodicZ == False:
        for y in [ -1, 1 ]:
            ot.pd.add_replication( [ 0, y, 0 ] )

    elif PeriodicX == True and PeriodicY == False and PeriodicZ == False:
        for x in [ -1, 1 ]:
            ot.pd.add_replication( [ x, 0, 0 ] )

    elif PeriodicX == True and PeriodicY == True and PeriodicZ == True:
        for x in [ -1, 0, 1 ]:
            for y in [ -1, 0, 1 ]:
                for z in [-1, 0, 1]:
                    if x or y or z:
                        ot.pd.add_replication( [ x, y, z ] )
    
    else:
        AssertionError('Please specify the periodicity')

    #print('Target masses before Damped Newton', ot.get_masses())
    #print('Weights before Damped Newton', ot.get_weights())
    #print('Mass before Damped Newton', ot.pd.integrals())

    ot.adjust_weights() #Use Damped Newton to find the optimal weight
    psi = ot.get_weights() #Extract the optimal weights from the solver

    #print('Mass after Damped Newton', ot.pd.integrals()) #Print the mass of each cell
    #print('Difference in initial and final weights', np.linalg.norm(psi0-psi)) #Check how different the initial guess is from the optimal weights

    return (laguerre_centroids(domain, Y, psi), psi)

