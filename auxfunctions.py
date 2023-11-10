import numpy as np
import sys
import csv
csv.field_size_limit(min(sys.maxsize, 2147483646))

def load_data(data):

    # Initialize lists to store the loaded data
    seeds = []
    centroids = []
    weights = []
    mass = []

    # Load the data from the CSV file
    with open(data, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            seeds.append(eval(row['Seeds']))
            centroids.append(eval(row['Centroids']))
            weights.append(eval(row['Weights']))
            mass.append(eval(row['Mass']))

    # Access the individual arrays
    Z = np.array(seeds)
    C = np.array(centroids)
    W = np.array(weights)
    M = np.array(mass)

    return Z, C, W, M


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
    
    if PeriodicX and PeriodicY and PeriodicZ:
        # Wrap points in x, y, and z components
        Z[:, 0] = (Z[:, 0] - box[0]) % (box[3] - box[0]) + box[0]
        Z[:, 1] = (Z[:, 1] - box[1]) % (box[4] - box[1]) + box[1]
        Z[:, 2] = (Z[:, 2] - box[2]) % (box[5] - box[2]) + box[2]
    elif PeriodicX and PeriodicY and not PeriodicZ:
        # Wrap points in the x and y component
        Z[:, 0] = (Z[:, 0] - box[0]) % (box[3] - box[0]) + box[0]
        Z[:, 1] = (Z[:, 1] - box[1]) % (box[4] - box[1]) + box[1]
    elif PeriodicX and not PeriodicY and PeriodicZ:
        # Wrap points in the x and z component
        Z[:, 0] = (Z[:, 0] - box[0]) % (box[3] - box[0]) + box[0]
        Z[:, 2] = (Z[:, 2] - box[2]) % (box[5] - box[2]) + box[2]
    elif not PeriodicX and PeriodicY and PeriodicZ:
        # Wrap points in the y and z component
        Z[:, 1] = (Z[:, 1] - box[1]) % (box[4] - box[1]) + box[1]
        Z[:, 2] = (Z[:, 2] - box[2]) % (box[5] - box[2]) + box[2]
    elif PeriodicX and not PeriodicY and not PeriodicZ:
        # Wrap points in the x component
        Z[:, 0] = (Z[:, 0] - box[0]) % (box[3] - box[0]) + box[0]
    elif not PeriodicX and PeriodicY and not PeriodicZ:
        # Wrap points in the y component
        Z[:, 1] = (Z[:, 1] - box[1]) % (box[4] - box[1]) + box[1]
    elif not PeriodicX and not PeriodicY and PeriodicZ:
        # Wrap points in the z component
        Z[:, 2] = (Z[:, 2] - box[2]) % (box[5] - box[2]) + box[2]
    
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

def cyc_temp_surface(x, y, z):
    return -(y/(1 + y**2)) - 0.12 * y + 0.15 * cyc_pertub(x, y)

def cyc_temp_lid(x, y, z, A):
    return -(1/2) * (y/((1 + z)**2 + y**2) + y/((1 - z)**2 + y**2)) - 0.12 * y + A * z - 0.6 * cyc_pertub(x + 1, y) + 1.29

def MerVel(Z, C, W):

    # Physical Constants
    f = 1e-4

    # Compute Meridonal Velocities
    MVel = [[0] * len(W[0]) for _ in range(len(W))]
    for i in range(len(W)):
        for j in range(len(W[0])):
            MVel[i][j] = f * (Z[i][j][0] - C[i][j][0])

    return MVel

def ZonVel(Z, C, W):

    # Physical Constants
    f = 1e-4

    # Compute Zonal Velocities
    ZVel = [[0] * len(W[0]) for _ in range(len(W))]
    for i in range(len(W)):
        for j in range(len(W[0])):
            ZVel[i][j] = f * ( - Z[i][j][1] + C[i][j][1])
        
    return ZVel