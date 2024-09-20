import numpy as np
from scipy import sparse
import scipy.sparse.linalg as spla
import optimaltransportsolver as ots
import petsc4py
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

    # Construct a matrix of perturbations
    perturbation = np.random.uniform(0.99, 1, size=(N, 3))

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

        snes = PETSc.SNES().create()
        snes.setType('newtonls')
        snes.setTolerances(max_it=100000, rtol=1e-5, atol=1e-5, stol=1e-2)
        ksp = snes.getKSP()
        ksp.setType('gmres')
        pc = ksp.getPC()
        pc.setType('gamg')

        for i in range(1, Ndt):
            if debug:
                print(f"Time Step {i}")

            Z_CN = PETSc.Vec().createWithArray(Z.flatten())
            f = PETSc.Vec().createWithArray(np.zeros_like(Z.flatten()))

            def residualSNES(snes, x, f):
                Z_CN_arr = x.getArray(readonly=True).reshape((N, 3)).copy()
                w0 = wg.rescale_weights(box, Z_CN_arr, np.zeros(shape=(N,)), PeriodicX, PeriodicY, PeriodicZ)[0]
                sol_CN = ots.ot_solve(D, Z_CN_arr, w0, err_tol, PeriodicX, PeriodicY, PeriodicZ, box, solver, False)

                F = (Z_CN_arr - Z).flatten() - (dt / 2) * (J.dot((Z_CN_arr - sol_CN[0]).flatten()) + J.dot((Z - sol[0]).flatten()))
                f.setArray(F)

            def formJacobian(snes, x, Jmat, Pmat):
                Z_CN_arr = x.getArray(readonly=True).reshape((N, 3)).copy()
                w0 = wg.rescale_weights(box, Z_CN_arr, np.zeros(shape=(N,)), PeriodicX, PeriodicY, PeriodicZ)[0]
                sol_CN = ots.ot_solve(D, Z_CN_arr, w0, err_tol, PeriodicX, PeriodicY, PeriodicZ, box, solver, False)

                m = sol_CN[4].copy()
                row_mask = np.delete(np.arange(m.shape[0]), np.arange(3, m.shape[0], 4))
                exclude_mask = (np.arange(4 * N) - 3) % 4 == 0
                column_mask = np.arange(4 * N)[~exclude_mask]
                DCDz = m[row_mask][:, column_mask]

                GradF = sparse.eye(3 * N) + (dt / 2) * J.dot(DCDz)
    
                print('Matrix norm', spla.norm(GradF))

                Jmat.setValuesCSR(GradF.indptr, GradF.indices, GradF.data)
                Jmat.assemble()

                if Pmat != Jmat:
                    Pmat.setValuesCSR(GradF.indptr, GradF.indices, GradF.data)
                    Pmat.assemble()

            def custom_monitor(snes, its, fgnorm):
                print(f"Iteration {its}, Residual norm {fgnorm}")

            Jmat = PETSc.Mat().create()
            Jmat.setSizes([3 * N, 3 * N])
            Jmat.setType('aij')
            Jmat.setUp()

            snes.setFunction(residualSNES, f)
            snes.setJacobian(formJacobian, Jmat, Jmat)

            # Add a custom monitor
            snes.setMonitor(custom_monitor)

            for i in range(1, Ndt):
                if debug:
                    print(f"Time Step {i}")

                Z_CN = PETSc.Vec().createWithArray(Z.flatten())
                f = PETSc.Vec().createWithArray(np.zeros_like(Z.flatten()))

                snes.solve(None, Z_CN)

                residualSNES(None, Z_CN, f)

                # Print preconditioner settings
                if snes.converged:
                    print(f"SNES converged at time step {i} with reason: {snes.getConvergedReason()} and final residual norm: {snes.getFunctionNorm()}")
                    Z_CN = Z_CN.getArray().reshape((N, 3))
                    w0 = wg.rescale_weights(box, Z_CN, np.zeros(shape=(N,)), PeriodicX, PeriodicY, PeriodicZ)[0]
                    sol_CN = ots.ot_solve(D, Z_CN, w0, err_tol, PeriodicX, PeriodicY, PeriodicZ, box, solver, False)

                    Zint = Z + (dt / 2) * (J.dot((Z_CN - sol_CN[0]).flatten()) + J.dot((Z - sol[0]).flatten())).reshape((N, 3))
                    Z = aux.get_remapped_seeds(box, Zint, PeriodicX, PeriodicY, PeriodicZ)

                    w0 = wg.rescale_weights(box, Z, np.zeros(shape=(N,)), PeriodicX, PeriodicY, PeriodicZ)[0]
                    sol = ots.ot_solve(D, Z, w0, err_tol, PeriodicX, PeriodicY, PeriodicZ, box, solver, False)

                    with open('./data/CN_SG_data.msgpack', 'ab') as msgpackfile:
                        msgpackfile.write(msgpack.packb({
                            'time_step': i,
                            'Seeds': Z.tolist(),
                            'Centroids': sol[0].tolist(),
                            'Weights': sol[1].tolist(),
                            'Mass': sol[2].tolist(),
                            'TransportCost': sol[3].tolist(),
                        }))
                else:
                    print("SNES did not converge at time step", i)
                    break