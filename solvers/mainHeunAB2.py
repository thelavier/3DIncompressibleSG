import numpy as np
from scipy import sparse
import optimaltransportsolver as ots
import weightguess as wg
import auxfunctions as aux
import msgpack
import os

def SG_solver(box, Z0, PercentTolerance, FinalTime, Ndt, PeriodicX, PeriodicY, PeriodicZ, solver = 'Petsc', debug = False):
    """
    Solves the Semi-Geostrophic equations using the geometric method.

    Parameters:
    - box (list or tuple): Domain bounds [xmin, ymin, zmin, xmax, ymax, zmax].
    - Z0 (array): Initial seed positions.
    - PercentTolerance (float): Percentage tolerance (e.g., 1 for 1% tolerance).
    - FinalTime (float): Endpoint of the simulation time.
    - Ndt (int): Number of steps from t=0 to FinalTime.
    - PeriodicX, PeriodicY, PeriodicZ (bool): Indicates if the boundaries are periodic in x, y, z respectively.
    - solver (str): Indicates the linear solver to use ('Petsc' or 'Scipy').
    - debug (bool): Enables debugging mode.

    Returns:
    - None: Saves the seed positions, centroid positions, and optimal weights at every timestep in a MessagePack file.
    """
    # Setup and initialization
    N = len(Z0)
    dt = FinalTime / Ndt
    D = ots.make_domain(box, PeriodicX, PeriodicY, PeriodicZ) # Construct the domain
    Lx, Ly, Lz = [box[i+3] - box[i] for i in range(3)]
    err_tol = (PercentTolerance / 100) * (Lx * Ly * Lz / N)

    # Setup extended J matrix for RHS of the ODE
    P = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
    J = sparse.kron(sparse.eye(N, dtype=int), sparse.csr_matrix(P))

    # Delete the MessagePack file if it exists to start fresh
    if os.path.exists('./data/HeunAB2_SG_data.msgpack'):
        os.remove('./data/HeunAB2_SG_data.msgpack')

    # Open the MessagePack file for writing
    with open('./data/HeunAB2_SG_data.msgpack', 'wb') as msgpackfile:
        # Define the header data
        header_data = {
            'fieldnames': ['time_step', 'Seeds', 'Centroids', 'Weights', 'Mass', 'TransportCost'],
        }

        # Write the header using MessagePack
        msgpackfile.write(msgpack.packb(header_data))

    # Open the MessagePack file for writing and write the header
    with open('./data/HeunAB2_SG_data.msgpack', 'ab') as msgpackfile:
        
        if debug:
            print("Time Step 0") # Use for tracking progress of the code when debugging.

        # Construct the initial state
        w0 = wg.rescale_weights(box, Z0, np.zeros(shape = (N,)), PeriodicX, PeriodicY, PeriodicZ)[0] # Rescale the weights to generate an optimized initial guess
        sol = ots.ot_solve(D, Z0, w0, err_tol, PeriodicX, PeriodicY, PeriodicZ, box, solver, debug) # Solve the optimal transport problem

        # Create a sliding window buffer for Z, C, w, and M
        Z_window = [Z0.copy(), Z0.copy(), Z0.copy()]
        C_window = [sol[0].copy(), sol[0].copy(), sol[0].copy()]
        w_window = [sol[1].copy(), sol[1].copy(), sol[1].copy()]
        m_window = [sol[2].copy(), sol[2].copy(), sol[2].copy()]
        TC_window = [sol[3].copy(), sol[3].copy(), sol[3].copy()]

        if debug:
            print("Time Step 1") # Use for tracking progress of the code when debugging.

        # Predictor Step with Forward Euler
        Z_pred = Z_window[0] + dt * J.dot(np.array(Z_window[0] - C_window[0]).flatten()).reshape((N, 3)) # Use forward Euler
        Z_pred = aux.get_remapped_seeds(box, Z_pred, PeriodicX, PeriodicY, PeriodicZ)

        w_pred = wg.rescale_weights(box, Z_pred, np.zeros(shape = (N,)), PeriodicX, PeriodicY, PeriodicZ)[0] # Rescale the weights to generate an optimized initial guess
        sol_pred = ots.ot_solve(D, Z_pred, w_pred, err_tol, PeriodicX, PeriodicY, PeriodicZ, box, solver, debug) # Solve the optimal transport problem

        # Corrector step (average the slopes at the original and predicted positions)
        Z_corr = Z_window[0] + (dt / 2) * (J.dot(np.array(Z_window[0] - C_window[0]).flatten()) + J.dot(np.array(Z_pred - sol_pred[0]).flatten())).reshape((N, 3)) # Use Heun
        Z_window[1] = aux.get_remapped_seeds(box, Z_corr, PeriodicX, PeriodicY, PeriodicZ)

        # Rescale the weights and solve the optimal transport problem
        w0 = wg.rescale_weights(box, Z_window[1], np.zeros(shape = (N,)), PeriodicX, PeriodicY, PeriodicZ)[0]
        sol = ots.ot_solve(D, Z_window[1], w0, err_tol, PeriodicX, PeriodicY, PeriodicZ, box, solver, debug)

        C_window[1] = sol[0].copy() # Store the centroids
        w_window[1] = sol[1].copy() # Store the optimal weights
        m_window[1] = sol[2].copy() # Store the mass of each cell
        TC_window[1] = sol[3].copy() # Store the transport cost of each cell

        # Save the data for time step 0 and 1
        msgpackfile.write(msgpack.packb({
            'time_step': 0,
            'Seeds': Z_window[0].tolist(),
            'Centroids': C_window[0].tolist(),
            'Weights': w_window[0].tolist(),
            'Mass': m_window[0].tolist(),
            'TransportCost': TC_window[0].tolist(),
        }))

        msgpackfile.write(msgpack.packb({
            'time_step': 1,
            'Seeds': Z_window[1].tolist(),
            'Centroids': C_window[1].tolist(),
            'Weights': w_window[1].tolist(),
            'Mass': m_window[1].tolist(),
            'TransportCost': TC_window[1].tolist(),
        }))

        # Apply Adams-Bashforth 2 to solve the ODE
        for i in range(2, Ndt):

            if debug:
                print(f"Time Step {i}") # Use for tracking progress of the code when debugging

            # Use Adams-Bashforth to take a predictor step time step
            Z_pred = Z_window[(i - 1) % 3] + (dt / 2) * (3 * J.dot(np.array(Z_window[(i - 1) % 3] - C_window[(i - 1) % 3]).flatten()) - J.dot(np.array(Z_window[(i - 2) % 3] - C_window[(i - 2) % 3]).flatten())).reshape((N, 3)) # Use AB2
            Z_pred = aux.get_remapped_seeds(box, Z_pred, PeriodicX, PeriodicY, PeriodicZ)

            w_pred = wg.rescale_weights(box, Z_pred, np.zeros(shape = (N,)), PeriodicX, PeriodicY, PeriodicZ)[0]
            sol = ots.ot_solve(D, Z_pred, w_pred, err_tol, PeriodicX, PeriodicY, PeriodicZ, box, solver, debug)

            # Corrector step (average the slopes at the original and predicted positions)
            Z_corr = Z_window[(i - 1) % 3] + (dt / 2) * (J.dot(np.array(Z_window[(i - 1) % 3] - C_window[(i - 1) % 3]).flatten()) + J.dot(np.array(Z_pred - sol_pred[0]).flatten())).reshape((N, 3)) # Use Heun
            Z_window[i % 3] = aux.get_remapped_seeds(box, Z_corr, PeriodicX, PeriodicY, PeriodicZ)

            # Rescale the weights and solve the optimal transport problem
            w0 = wg.rescale_weights(box, Z_window[i % 3], np.zeros(shape = (N,)), PeriodicX, PeriodicY, PeriodicZ)[0]
            sol = ots.ot_solve(D, Z_window[i % 3], w0, err_tol, PeriodicX, PeriodicY, PeriodicZ, box, solver, debug)

            # Save the centroids and optimal weights
            C_window[i % 3] = sol[0].copy()
            w_window[i % 3] = sol[1].copy()
            m_window[i % 3] = sol[2].copy()
            TC_window[i % 3] = sol[3].copy() 

            # Save the data for Z, C, w, and M continuously
            msgpackfile.write(msgpack.packb({
                'time_step': i,
                'Seeds': Z_window[i % 3].tolist(),
                'Centroids': C_window[i % 3].tolist(),
                'Weights': w_window[i % 3].tolist(),
                'Mass': m_window[i % 3].tolist(),
                'TransportCost': TC_window[i % 3].tolist(),
            }))
