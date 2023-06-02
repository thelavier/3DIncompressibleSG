import numpy as np

def get_remapped_seeds(box, Z, PeriodicX, PeriodicY, PeriodicZ):
    """
    A function that remaps the seeds so that they remain in the periodic domain

    Inputs:
        box: the fluid domain given as a list [xmin, ymin, zmin, x max, ymax, zmax]
        Z: the seed positions
        PeriodicX: a boolian specifying periodicity in x
        PeriodicY: a boolian specifying periodicity in y
        PeriodicZ: a boolian specifying periodicity in z

    Outputs:
        Z: the seeds remaped to be inside the fluid domain
    """
    
    p = [PeriodicX, PeriodicY, PeriodicZ]

    bxDims = [box[3] - box[0], box[4] - box[1], box[5] - box[2]] # Dimensions of the domain
    Binv = np.diag(p) / np.array(bxDims) # Create a 3x3 matrix to normalize the seeds in the periodic domain

    k = np.floor(-np.dot(Z, Binv) + 0.5 * np.ones((1, 3)))

    Z = Z + np.dot(k, np.diag(bxDims))

    return Z