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

def get_point_transform(point, matrix):
    """
    Inputs:
        point: 
        matrix:

    Outputs:
        transformed_point:
    """
    # Convert corner to a 1x3 matrix
    point_matrix = np.array(point).reshape(3, 1)
    
    # Perform the transformation
    transformed_point0 = np.dot(matrix, point_matrix)
    
    # Convert back to a list
    transformed_point = transformed_point0.flatten().tolist()
    
    return transformed_point

def cyc_pertub(x, y):
    return (1 + (x/0.5)**2 + (y/0.5)**2)**(-(3/2)) - (1/2) * ((1 + ((x - 1)/0.5)**2 + (y/0.5)**2)**(-(3/2)) + (1 + ((x + 1)/0.5)**2 + (y/0.5)**2)**(-(3/2)))

def cyc_perturb_surface(row):
    return -(row[1]/(1 + row[1]**2)) - 0.12 * row[1] + 0.15 * cyc_pertub(row[0], row[1])

def cyc_perturb_lid(row, A):
    return -(1/2) * (row[1]/((1 + 0.45)**2 + row[1]**2) + row[1]/((1 - 0.45)**2 + row[1]**2)) - 0.12 * row[1] + 0.45*A - 0.6 * cyc_pertub(row[0] + 1, row[1])

def cyc_pertub_body(row, A):
    return -(1/2) * (row[1]/((1 + row[0])**2 + row[1]**2) + row[1]/((1 - row[0])**2 + row[1]**2)) - 0.12 * row[1] + A * row[0]