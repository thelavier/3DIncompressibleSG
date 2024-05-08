import numpy as np
from scipy import sparse
import optimaltransportsolver as ots
from petsc4py import PETSc
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
    maxiter = 10
    newttol = 1e-3

    # Setup extended J matrix for RHS of the ODE
    P = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
    J = sparse.kron(sparse.eye(N, dtype=int), sparse.csr_matrix(P))

    # Delete the MessagePack file if it exists to start fresh
    if os.path.exists('./data/CN_SG_data.msgpack'):
        os.remove('./data/CN_SG_data.msgpack')

    # Open the MessagePack file for writing
    with open('./data/CN_SG_data.msgpack', 'wb') as msgpackfile:
        # Define the header data
        header_data = {
            'fieldnames': ['time_step', 'Seeds', 'Centroids', 'Weights', 'Mass', 'TransportCost'],
        }

        # Write the header using MessagePack
        msgpackfile.write(msgpack.packb(header_data))

    # Open the MessagePack file for writing and write the header
    with open('./data/CN_SG_data.msgpack', 'ab') as msgpackfile:

        if debug:
            print("Time Step 0")
        
        # Construct the initial state
        Z = Z0.copy() 
        w0 = wg.rescale_weights(box, Z, np.zeros(shape = (N,)), PeriodicX, PeriodicY, PeriodicZ)[0] # Rescale the weights to generate an optimized initial guess
        sol = ots.ot_solve(D, Z, w0, err_tol, PeriodicX, PeriodicY, PeriodicZ, box, solver, debug) # Solve the optimal transport problem

        # Save the data for time step 0
        msgpackfile.write(msgpack.packb({
            'time_step': 0,
            'Seeds': Z.tolist(),
            'Centroids': sol[0].tolist(),
            'Weights': sol[1].tolist(),
            'Mass': sol[2].tolist(),
            'TransportCost': sol[3].tolist(),
        }))
        
        # Apply CN's Method to solve the ODE
        for i in range(1, Ndt):

            if debug:
                print(f"Time Step {i}") # Use for tracking progress of the code when debugging

            # Set initial guess for Newton-Raphson
            Z_CN = Z

            # Newton-Raphson Iterations
            for newtiter in range(maxiter):

                # Before the Newton-Raphson iterations start
                if newtiter == 0:
                    sol_CN = sol # Use the previously solved optimal transport problem's solution
                else:
                    # Rescale the weights and solve the optimal transport problem
                    w0 = wg.rescale_weights(box, Z_CN, np.zeros(shape = (N,)), PeriodicX, PeriodicY, PeriodicZ)[0]
                    sol_CN = ots.ot_solve(D, Z_CN, w0, err_tol, PeriodicX, PeriodicY, PeriodicZ, box, solver, debug)

                F = (Z_CN - Z).flatten() - (dt / 2) * (J.dot((Z_CN - sol_CN[0]).flatten()) + J.dot((Z - sol[0]).flatten()))

                # Construct DCDz 
                m = sol_CN[4].copy()
                row_mask = np.delete(np.arange(m.shape[0]), np.arange(3, m.shape[0], 4)) # Delete every 4th row
                exclude_mask = (np.arange(4 * N) - 3) % 4 == 0 # Create an exclude mask for the columns
                column_mask = np.arange(4*N)[~exclude_mask] # Invert the mask to delete every 4th column
                DCDz = m[row_mask][:, column_mask]

                # Construct the Gradient of F
                GradF = sparse.eye(3 * N) + (dt / 2) * J.dot(DCDz)

                # Convert GradF to PETSc matrix for the solver
                A = PETSc.Mat().createAIJ(size=GradF.shape, csr=(GradF.indptr, GradF.indices, GradF.data))
                b = PETSc.Vec().createWithArray(F)  # Assuming F is a 1-D array
        
                # Solve the linear system Ax = B using PETSc solver
                delta_Z = aux.solve(A, b)

                # Check for convergence
                if np.linalg.norm(delta_Z.getArray()) < newttol:
                    break # Convergence achieved, use Z_CN as it is for the Crank-Nicolson step
                else:
                    # Update Z_CN for the next iteration
                    Z_CN -= aux.get_remapped_seeds(box, np.array(delta_Z).reshape((N, 3)), PeriodicX, PeriodicY, PeriodicZ)

            # Crank-Nicolson step (average the slopes at the original and predicted positions)
            Zint = Z + (dt / 2) * (J.dot(np.array(Z_CN - sol_CN[0]).flatten()) + J.dot(np.array(Z - sol[0]).flatten())).reshape((N, 3)) # Use Crank-Nicolson
            Z = aux.get_remapped_seeds(box, Zint, PeriodicX, PeriodicY, PeriodicZ)

            # Rescale the weights and solve the optimal transport problem
            w0 = wg.rescale_weights(box, Z, np.zeros(shape = (N,)), PeriodicX, PeriodicY, PeriodicZ)[0]
            sol = ots.ot_solve(D, Z, w0, err_tol, PeriodicX, PeriodicY, PeriodicZ, box, solver, debug)

            # Save the data continuously
            msgpackfile.write(msgpack.packb({
                'time_step': i,
                'Seeds': Z.tolist(),
                'Centroids': sol[0].tolist(),
                'Weights': sol[1].tolist(),
                'Mass': sol[2].tolist(),
                'TransportCost': sol[3].tolist(),
            }))
