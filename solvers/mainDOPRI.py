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
    tol_abs = 1e-12
    tol_rel = 1e-6

    # Construct a matrix of perturbations
    perturbation = np.random.uniform(0.99, 1, size=(N, 3))

    # Setup extended J matrix for RHS of the ODE
    P = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
    J = sparse.kron(sparse.eye(N, dtype=int), sparse.csr_matrix(P))

    # Coefficients for Dormand-Prince
    b21 = 1/5
    b31, b32 = 3/40, 9/40
    b41, b42, b43 = 44/45, -56/15, 32/9
    b51, b52, b53, b54 = 19372/6561, -25360/2187, 64448/6561, -212/729
    b61, b62, b63, b64, b65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
    b71, b72, b73, b74, b75, b76 = 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84

    # Fourth and fifth order coefficients for final estimation
    c4 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    c5 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]

    # Delete the MessagePack file if it exists to start fresh
    if os.path.exists('./data/DOPRI_SG_data.msgpack'):
        os.remove('./data/DOPRI_SG_data.msgpack')

    # Open the MessagePack file for writing
    with open('./data/DOPRI_SG_data.msgpack', 'wb') as msgpackfile:
        # Define the header data
        header_data = {
            'fieldnames': ['time_step', 'Seeds', 'Centroids', 'Weights', 'Mass', 'TransportCost'],
        }

        # Write the header using MessagePack
        msgpackfile.write(msgpack.packb(header_data))

    # Open the MessagePack file for writing and write the header
    with open('./data/DOPRI_SG_data.msgpack', 'ab') as msgpackfile:

        if debug:
            print("Time Step 0")
        
        # Construct the initial state
        Z = Z0.copy() 
        w0 = wg.rescale_weights(box, Z * perturbation, np.zeros(shape = (N,)), PeriodicX, PeriodicY, PeriodicZ)[0] # Rescale the weights to generate an optimized initial guess
        sol = ots.ot_solve(D, Z * perturbation, w0, err_tol, PeriodicX, PeriodicY, PeriodicZ, box, solver, debug) # Solve the optimal transport problem

        # Save the data for time step 0
        msgpackfile.write(msgpack.packb({
            'time_step': 0,
            'Seeds': Z.tolist(),
            'Centroids': sol[0].tolist(),
            'Weights': sol[1].tolist(),
            'Mass': sol[2].tolist(),
            'TransportCost': sol[3].tolist(),
        }))
        
        # Initialize the time
        t = 0.0 

        # Apply DOPRI Method to solve the ODE
        while t < FinalTime:
            success = False
            while not success:
                if debug:
                    print(f"Current time: {t}, current dt: {dt}")

                # k1: Slope at the current state
                k1 = J.dot(np.array(Z - sol[0]).flatten()).reshape((N, 3))

                # Intermediate state for k2
                Z_k2 = Z + dt * b21 * k1
                Z_k2 = aux.get_remapped_seeds(box, Z_k2, PeriodicX, PeriodicY, PeriodicZ)
                w_k2 = wg.rescale_weights(box, Z_k2, np.zeros(shape=(N,)), PeriodicX, PeriodicY, PeriodicZ)[0]
                sol_k2 = ots.ot_solve(D, Z_k2, w_k2, err_tol, PeriodicX, PeriodicY, PeriodicZ, box, solver, debug)
                k2 = J.dot(np.array(Z_k2 - sol_k2[0]).flatten()).reshape((N, 3))
    
                # Intermediate state for k3
                Z_k3 = Z + dt * (b31 * k1 + b32 * k2)
                Z_k3 = aux.get_remapped_seeds(box, Z_k3, PeriodicX, PeriodicY, PeriodicZ)
                w_k3 = wg.rescale_weights(box, Z_k3, np.zeros(shape=(N,)), PeriodicX, PeriodicY, PeriodicZ)[0]
                sol_k3 = ots.ot_solve(D, Z_k3, w_k3, err_tol, PeriodicX, PeriodicY, PeriodicZ, box, solver, debug)
                k3 = J.dot(np.array(Z_k3 - sol_k3[0]).flatten()).reshape((N, 3))
    
                # Intermediate state for k4
                Z_k4 = Z + dt * (b41 * k1 + b42 * k2 + b43 * k3)
                Z_k4 = aux.get_remapped_seeds(box, Z_k4, PeriodicX, PeriodicY, PeriodicZ)
                w_k4 = wg.rescale_weights(box, Z_k4, np.zeros(shape=(N,)), PeriodicX, PeriodicY, PeriodicZ)[0]
                sol_k4 = ots.ot_solve(D, Z_k4, w_k4, err_tol, PeriodicX, PeriodicY, PeriodicZ, box, solver, debug)
                k4 = J.dot(np.array(Z_k4 - sol_k4[0]).flatten()).reshape((N, 3))

                # Intermediate state for k5
                Z_k5 = Z + dt * (b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4)
                Z_k5 = aux.get_remapped_seeds(box, Z_k5, PeriodicX, PeriodicY, PeriodicZ)
                w_k5 = wg.rescale_weights(box, Z_k5, np.zeros(shape=(N,)), PeriodicX, PeriodicY, PeriodicZ)[0]
                sol_k5 = ots.ot_solve(D, Z_k5, w_k5, err_tol, PeriodicX, PeriodicY, PeriodicZ, box, solver, debug)
                k5 = J.dot(np.array(Z_k5 - sol_k5[0]).flatten()).reshape((N, 3))

                # Intermediate state for k6
                Z_k6 = Z + dt * (b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5)
                Z_k6 = aux.get_remapped_seeds(box, Z_k6, PeriodicX, PeriodicY, PeriodicZ)
                w_k6 = wg.rescale_weights(box, Z_k6, np.zeros(shape=(N,)), PeriodicX, PeriodicY, PeriodicZ)[0]
                sol_k6 = ots.ot_solve(D, Z_k6, w_k6, err_tol, PeriodicX, PeriodicY, PeriodicZ, box, solver, debug)
                k6 = J.dot(np.array(Z_k6 - sol_k6[0]).flatten()).reshape((N, 3))

                # Final state for k7
                Z_k7 = Z + dt * (b71 * k1 + b72 * k2 + b73 * k3 + b74 * k4 + b75 * k5 + b76 * k6)
                Z_k7 = aux.get_remapped_seeds(box, Z_k7, PeriodicX, PeriodicY, PeriodicZ)
                w_k7 = wg.rescale_weights(box, Z_k7, np.zeros(shape=(N,)), PeriodicX, PeriodicY, PeriodicZ)[0]
                sol_k7 = ots.ot_solve(D, Z_k7, w_k7, err_tol, PeriodicX, PeriodicY, PeriodicZ, box, solver, debug)
                k7 = J.dot(np.array(Z_k7 - sol_k7[0]).flatten()).reshape((N, 3))
    
                # Estimate next state with both fourth-order and fifth-order
                Z_next_higher = Z + dt * sum(c5[i] * k for i, k in enumerate([k1, k2, k3, k4, k5, k6, k7]))
                Z_next_lower = Z + dt * sum(c4[i] * k for i, k in enumerate([k1, k2, k3, k4, k5, k6, k7]))

                # Calculate error
                error_estimate = np.linalg.norm(Z_next_higher - Z_next_lower)
                error_allowed = tol_abs + tol_rel * max(np.linalg.norm(Z), np.linalg.norm(Z_next_higher))
        
                # Check error against tolerance and adjust dt
                if error_estimate <= error_allowed:
                    if debug:
                        print(f"Step accepted. New time: {t}, new dt: {dt}")

                    # Accept step
                    Z = Z_next_higher
                    t += dt
                    success = True

                    # Solve the optimal transport problem for the updated Z
                    w0 = wg.rescale_weights(box, Z, np.zeros(shape=(N,)), PeriodicX, PeriodicY, PeriodicZ)[0]
                    sol = ots.ot_solve(D, Z, w0, err_tol, PeriodicX, PeriodicY, PeriodicZ, box, solver, debug)

                    # Optionally increase dt for the next step
                    dt *= (error_allowed / error_estimate) ** 0.25

                    # Save the data continuously
                    msgpackfile.write(msgpack.packb({
                        'time_step': round(t,5),
                        'Seeds': Z.tolist(),
                        'Centroids': sol[0].tolist(),
                        'Weights': sol[1].tolist(),
                        'Mass': sol[2].tolist(),
                        'TransportCost': sol[3].tolist(),
                    }))

                else:
                    # Reduce dt and retry
                    dt *= (error_allowed / error_estimate) ** 0.25
                    if debug:
                        print(f"Step rejected. Reducing dt to {dt}")