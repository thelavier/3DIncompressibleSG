import numpy as np

#Define the rescaling function to improve the inital guess for the damped Newton Solver
def Rescale_weights(bx, Z, psi, PeriodicX, PeriodicY, PeriodicZ):
    """
    Function returning weights w such that all cells in the 3d Laguerre tessellation of bx generated by (Z,w) have positive area.

    Inputs:
        bx: list or tuple defining domain [xmin, ymin, zmin, xmax, ymax, zmax]
        Z: numpy array of shape (n, 3) containing seed points
        psi: a guess at what the weights should be in the scaled box. 0 would be a Voronoi tessellation
        PeriodicX: a boolian indicating if the boundaries are periodic in x 
        PeriodicY: a boolian indicating if the boundaries are periodic in y
        PeriodicZ: a boolian indicating if the boundaries are periodic in z

    Outputs:
        w: numpy array of shape (n,) containing weight vector
        lambda_: scaling used when defining weights
        t: numpy array of shape (3,) containing translation used when defining weights
    """
    if PeriodicX == False and PeriodicY == False and PeriodicZ == False:

        # Define rescaling so that rescaled seeds lie in a translated copy of bx
        Z_x = Z[:, 0]
        min_Z_x = np.min(Z_x)
        max_Z_x = np.max(Z_x)

        Z_y = Z[:, 1]
        min_Z_y = np.min(Z_y)
        max_Z_y = np.max(Z_y)

        Z_z = Z[:, 2]
        min_Z_z = np.min(Z_z)
        max_Z_z = np.max(Z_z)

        lambda_x = (bx[3] - bx[0]) / (max_Z_x - min_Z_x)
        lambda_y = (bx[4] - bx[1]) / (max_Z_y - min_Z_y)
        lambda_z = (bx[5] - bx[2]) / (max_Z_z - min_Z_z)

        lambda_ = min(lambda_x, lambda_y, lambda_z) * (1 - 1e-2)

        # Define translation to be the center of the domain minus the center of the rescaled seeds
        c_dom = [(bx[3] + bx[0]) / 2, (bx[4] + bx[1]) / 2, (bx[5] + bx[2]) / 2]
        c_zl = [lambda_ * (min_Z_x + max_Z_x) / 2, lambda_ * (min_Z_y + max_Z_y) / 2, lambda_ * (min_Z_z + max_Z_z) / 2]
        t = np.array(c_dom) - np.array(c_zl)

        # Define weights
        w = (1 - lambda_) * np.square(np.linalg.norm(Z, axis = 1)) - 2 * np.dot(Z, t) - psi / lambda_

    elif PeriodicX == True and PeriodicY == True and PeriodicZ == False:

        # Define rescaling so that rescaled seeds lie in a translated copy of bx
        Z_z = Z[:, 2]
        min_Z_z = np.min(Z_z)
        max_Z_z = np.max(Z_z)

        lambda_ = (bx[5] - bx[2]) / (max_Z_z - min_Z_z) * (1 - 1e-2)

        # Define translation to be the center of the domain minus the center of the rescaled seeds
        c_dom = (bx[5] + bx[2]) / 2
        c_zl = lambda_ * (min_Z_z + max_Z_z) / 2
        t = c_dom - c_zl

        # Define weights
        w = (1 - lambda_) * np.square(np.linalg.norm(Z_z)) - 2 * Z_z * t - psi / lambda_


    elif PeriodicX == True and PeriodicY == False and PeriodicZ == True:

        # Define rescaling so that rescaled seeds lie in a translated copy of bx
        Z_y = Z[:, 1]
        min_Z_y = np.min(Z_y)
        max_Z_y = np.max(Z_y)

        lambda_ = (bx[4] - bx[1]) / (max_Z_y - min_Z_y) * (1 - 1e-2)

        # Define translation to be the center of the domain minus the center of the rescaled seeds
        c_dom = (bx[4] + bx[1]) / 2
        c_zl = lambda_ * (min_Z_y + max_Z_y) / 2
        t = c_dom - c_zl

        # Define weights
        w = (1 - lambda_) * np.square(np.linalg.norm(Z_y)) - 2 * Z_y * t - psi / lambda_

    elif PeriodicX == False and PeriodicY == True and PeriodicZ == True:

        # Define rescaling so that rescaled seeds lie in a translated copy of bx
        Z_x = Z[:, 0]
        min_Z_x = np.min(Z_x)
        max_Z_x = np.max(Z_x)

        lambda_ = (bx[3] - bx[0]) / (max_Z_x - min_Z_x) * (1 - 1e-2)

        # Define translation to be the center of the domain minus the center of the rescaled seeds
        c_dom = (bx[3] + bx[0]) / 2
        c_zl = lambda_ * (min_Z_x + max_Z_x) / 2
        t = c_dom - c_zl

        # Define weights
        w = (1 - lambda_) * np.square(np.linalg.norm(Z_x)) - 2 * Z_x * t - psi / lambda_

    elif PeriodicX == False and PeriodicY == False and PeriodicZ == True:

        # Define rescaling so that rescaled seeds lie in a translated copy of bx
        Z_x = Z[:, 0]
        min_Z_x = np.min(Z_x)
        max_Z_x = np.max(Z_x)

        Z_y = Z[:, 1]
        min_Z_y = np.min(Z_y)
        max_Z_y = np.max(Z_y)

        lambda_x = (bx[3] - bx[0]) / (max_Z_x - min_Z_x)
        lambda_y = (bx[4] - bx[1]) / (max_Z_y - min_Z_y)

        lambda_ = min(lambda_x, lambda_y) * (1 - 1e-2)

        # Define translation to be the center of the domain minus the center of the rescaled seeds
        c_dom = [(bx[3] + bx[0]) / 2, (bx[4] + bx[1]) / 2]
        c_zl = [lambda_ * (min_Z_x + max_Z_x) / 2, lambda_ * (min_Z_y + max_Z_y) / 2]
        t = np.array(c_dom) - np.array(c_zl)

        # Define weights NEED TO CORRECT THIS
        w = (1 - lambda_) * np.square(np.linalg.norm(Z, axis = 1)) - 2 * np.dot(Z, t) - psi / lambda_

    elif PeriodicX == False and PeriodicY == True and PeriodicZ == False:

        # Define rescaling so that rescaled seeds lie in a translated copy of bx
        Z_x = Z[:, 0]
        min_Z_x = np.min(Z_x)
        max_Z_x = np.max(Z_x)

        Z_z = Z[:, 2]
        min_Z_z = np.min(Z_z)
        max_Z_z = np.max(Z_z)

        lambda_x = (bx[3] - bx[0]) / (max_Z_x - min_Z_x)
        lambda_z = (bx[5] - bx[2]) / (max_Z_z - min_Z_z)

        lambda_ = min(lambda_x, lambda_z) * (1 - 1e-2)

        # Define translation to be the center of the domain minus the center of the rescaled seeds
        c_dom = [(bx[3] + bx[0]) / 2, (bx[5] + bx[2]) / 2]
        c_zl = [lambda_ * (min_Z_x + max_Z_x) / 2, lambda_ * (min_Z_z + max_Z_z) / 2]
        t = np.array(c_dom) - np.array(c_zl)

        # Define weights NEED TO CORRECT THIS
        w = (1 - lambda_) * np.square(np.linalg.norm(Z, axis = 1)) - 2 * np.dot(Z, t) - psi / lambda_

    elif PeriodicX == True and PeriodicY == False and PeriodicZ == False:

        # Define rescaling so that rescaled seeds lie in a translated copy of bx
        Z_y = Z[:, 1]
        min_Z_y = np.min(Z_y)
        max_Z_y = np.max(Z_y)

        Z_z = Z[:, 2]
        min_Z_z = np.min(Z_z)
        max_Z_z = np.max(Z_z)

        lambda_y = (bx[4] - bx[1]) / (max_Z_y - min_Z_y)
        lambda_z = (bx[5] - bx[2]) / (max_Z_z - min_Z_z)

        lambda_ = min(lambda_y, lambda_z) * (1 - 1e-2)

        # Define translation to be the center of the domain minus the center of the rescaled seeds
        c_dom = [(bx[4] + bx[1]) / 2, (bx[5] + bx[2]) / 2]
        c_zl = [lambda_ * (min_Z_y + max_Z_y) / 2, lambda_ * (min_Z_z + max_Z_z) / 2]
        t = np.array(c_dom) - np.array(c_zl)

        # Define weights NEED TO FIX THIS
        w = (1 - lambda_) * np.square(np.linalg.norm(Z, axis = 1)) - 2 * np.dot(Z, t) - psi / lambda_

    elif PeriodicX == True and PeriodicY == True and PeriodicZ == True:

        w = psi
        lambda_ = 0
        t = 0
    
    else:
        AssertionError('Please specify the periodicity')

    return w, lambda_, t