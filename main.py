import numpy as np
import AuxFunctions as aux

def SGSolver(Box, InitialSeeds, NumberofSeeds, PercentTolerance, FinalTime, NumberofSteps):
    """
    Function solving the Semi-Geostrophic equations using the geometric method.

    Inputs:
        box: list or tuple defining domain [xmin, ymin, zmin, xmax, ymax, zmax]
        InitialSeeds: The intial seed positions 
        NumberofSeeds: The number of seeds
        PercentTolerance: Percent tolerance, ex. 1 means 1% tolerance
        FinalTime: The end point of the simulation
        NumberofSteps: The number of steps to take to get from t=0 to t=time final

        Note: The last two parameters are set up this way to integrate more easily with the animator, could be changed 

    Outputs:
        data: Outputs a saved datafile that contains the seed positions, centroid positions, and optimal weights at every timestep
    """
    #Bring parameters into the function
    box = Box
    Z = InitialSeeds
    N = NumberofSeeds
    per_tol = PercentTolerance
    tf = FinalTime
    Ndt = NumberofSteps

    #Construct the domain
    D = aux.make_domain(box)

    #Compute the stepsize
    dt = tf/Ndt

    #Setup extended J matrix for RHS of the ODE
    P = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
    J = np.kron(np.eye(N, dtype=int), P)

    #Setup empty data structure
    dataZ = np.zeros((Ndt, N, 3)) 
    dataC = np.zeros((Ndt, N, 3))
    dataw = np.zeros((Ndt, N))

    #Establish solver parameters
    t = 0
    err_tol = ( per_tol / 100 ) * (D.measure() / N) #Build the relative error tollereance 

    #Apply forward Euler method to solve the ODE system
    for i in range(Ndt):

        #Rescale the weights to generate an optimized initial guess
        #w0 = 0*np.ones(N)
        w0 = aux.Rescale_weights(box, Z)[0]

        #Solve the optimal transport problem
        sol = aux.ot_centroids(D, Z, w0, err_tol)
        C = sol[0]

        #Use forward Euler to take a time step
        Z = Z+dt*(J@(np.array(Z-C).flatten())).reshape((N, 3))

        #Save the data
        dataZ[i] = Z.copy()
        dataC[i] = C.copy()
        dataw[i] = sol[1].copy()

        #Increment the solver parameters
        t = t + dt

    #Save the data
    np.savez('SG_data.npz', data1 = dataZ, data2 = dataC, data3 = dataw)
