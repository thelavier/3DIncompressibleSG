import numpy as np
import optimaltransportsolver as ots
import weightguess as wg
import auxfunctions as aux
import csv
import os

def SG_solver(Box, InitialSeeds, NumberofSeeds, PercentTolerance, FinalTime, NumberofSteps, PeriodicX, PeriodicY, PeriodicZ, solver = 'Petsc', debug = False):
    """
    Function solving the Semi-Geostrophic equations using the geometric method.

    Inputs:
        box: list or tuple defining domain [xmin, ymin, zmin, xmax, ymax, zmax]
        InitialSeeds: The intial seed positions 
        NumberofSeeds: The number of seeds
        PercentTolerance: Percent tolerance, ex. 1 means 1% tolerance
        FinalTime: The end point of the simulation
        NumberofSteps: The number of steps to take to get from t=0 to t=time final
        PeriodicX: a boolian indicating if the boundaries are periodic in x 
        PeriodicY: a boolian indicating if the boundaries are periodic in y
        PeriodicZ: a boolian indicating if the boundaries are periodic in z
        solver: a string indicating if the code is using the Petsc or the Scipy linear solver
        debug: a boolian to put the code into debugging mode

        Note: The last two parameters are set up this way to integrate more easily with the animator, could be changed 

    Outputs:
        data: Outputs a saved datafile that contains the seed positions, centroid positions, and optimal weights at every timestep
    """
    # Bring parameters into the function
    box = Box
    Z0 = InitialSeeds
    N = NumberofSeeds
    per_tol = PercentTolerance
    tf = FinalTime
    Ndt = NumberofSteps

    # Delete the CSV file if it exists to start fresh
    if os.path.exists('./data/SG_data.csv'):
        os.remove('./data/SG_data.csv')

    # Open the CSV file for writing and create the header
    with open('./data/SG_data.csv', 'w', newline='') as csvfile:
        fieldnames = ['time_step', 'Seeds', 'Centroids', 'Weights', 'Mass']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader() 

    # Construct the domain
    D = ots.make_domain(box, PeriodicX, PeriodicY, PeriodicZ)

    # Compute the stepsize
    dt = tf/Ndt

    # Setup extended J matrix for RHS of the ODE
    P = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
    J = np.kron(np.eye(N, dtype=int), P)

    # Build the relative error tollereance 
    err_tol = ( per_tol / 100 ) * (D.measure() / N) 

    if debug == True:
        print("Time Step", 0) # Use for tracking progress of the code when debugging.
    else:
        pass

    # Construct the initial state
    w0 = wg.rescale_weights(box, Z0, np.zeros(shape = (N,)), PeriodicX, PeriodicY, PeriodicZ)[0] # Rescale the weights to generate an optimized initial guess
    sol = ots.ot_solve(D, Z0, w0, err_tol, PeriodicX, PeriodicY, PeriodicZ, box, solver, debug) # Solve the optimal transport problem

    # Create a sliding window buffer for Z, C, w, and M
    Z_window = [Z0.copy(), Z0.copy(), Z0.copy()]
    C_window = [sol[0].copy(), sol[0].copy(), sol[0].copy()]
    w_window = [sol[1].copy(), sol[1].copy(), sol[1].copy()]
    m_window = [sol[2].copy(), sol[2].copy(), sol[2].copy()]

    if debug == True:
        print("Time Step", 1) # Use for tracking progress of the code when debugging.

    # Use forward Euler to take an initial time step
    Zint = Z_window[0] + dt * (J @ (np.array(Z_window[0] - C_window[0]).flatten())).reshape((N, 3))
    Z_window[1] = aux.get_remapped_seeds(box, Zint, PeriodicX, PeriodicY, PeriodicZ)

    w0 = wg.rescale_weights(box, Z_window[1], np.zeros(shape = (N,)), PeriodicX, PeriodicY, PeriodicZ)[0] # Rescale the weights to generate an optimized initial guess
    sol = ots.ot_solve(D, Z_window[1], w0, err_tol, PeriodicX, PeriodicY, PeriodicZ, box, solver, debug) # Solve the optimal transport problem

    C_window[1] = sol[0].copy() # Store the centroids
    w_window[1] = sol[1].copy() # Store the optimal weights
    m_window[1] = sol[2].copy() # Store the mass of each cell

    # Save the data for time step 0 and 1
    with open('./data/SG_data.csv', 'a', newline='') as csvfile:
        fieldnames = ['time_step', 'Seeds', 'Centroids', 'Weights', 'Mass']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({
            'time_step': 0,
            'Seeds': Z_window[0].tolist(),
            'Centroids': C_window[0].tolist(),
            'Weights': w_window[0].tolist(),
            'Mass': m_window[0].tolist(),
        })
        writer.writerow({
            'time_step': 1,
            'Seeds': Z_window[1].tolist(),
            'Centroids': C_window[1].tolist(),
            'Weights': w_window[1].tolist(),
            'Mass': m_window[1].tolist(),
        })

    # Apply Adams-Bashforth 2 to solve the ODE
    for i in range(2, Ndt):

        if debug == True:
            print(i) # Use for tracking progress of the code when debugging

        # Use Adams-Bashforth to take a time step
        Zint = Z_window[(i - 1) % 3] + (dt / 2) * (3 * J @ (np.array(Z_window[(i - 1) % 3] - C_window[(i - 1) % 3]).flatten()) - J @ (np.array(Z_window[(i - 2) % 3] - C_window[(i - 2) % 3]).flatten())).reshape((N, 3))
        Z_window[i % 3] = aux.get_remapped_seeds(box, Zint, PeriodicX, PeriodicY, PeriodicZ)

        # Rescale the weights to generate an optimized initial guess
        w0 = wg.rescale_weights(box, Z_window[i % 3], np.zeros(shape = (N,)), PeriodicX, PeriodicY, PeriodicZ)[0]

        # Solve the optimal transport problem
        sol = ots.ot_solve(D, Z_window[i % 3], w0, err_tol, PeriodicX, PeriodicY, PeriodicZ, box, solver, debug)

        # Save the centroids and optimal weights
        C_window[i % 3] = sol[0].copy()
        w_window[i % 3] = sol[1].copy()
        m_window[i % 3] = sol[2].copy()

        # Save the data for Z, C, and w continuously
        with open('./data/SG_data.csv', 'a', newline='') as csvfile:
            fieldnames = ['time_step', 'Seeds', 'Centroids', 'Weights', 'Mass']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'time_step': i,
                'Seeds': Z_window[i % 3].tolist(),
                'Centroids': C_window[i % 3].tolist(),
                'Weights': w_window[i % 3].tolist(),
                'Mass': m_window[i % 3].tolist(),
            })
