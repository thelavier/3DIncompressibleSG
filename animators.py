import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import auxfunctions as aux

def point_animator(data, ZorC, Dim, box, tf):
    """
    Animates the solution to the ODE using data from the optimal transport solver.

    Parameters:
        data (str): File name containing the data.
        ZorC (str): 'Z' to animate seeds, 'C' to animate weights.
        Dim (str): '2D' or '3D' for the animation dimension.
        box (list or tuple): Domain definition [xmin, ymin, zmin, xmax, ymax, zmax].
        tf (int): Final time for the solver to determine frame rate.

    Returns:
        Matplotlib animation: An animation of the seeds or centroids.
    """
    # Configure animation settings
    plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg'

    # Load data
    Z, C, _, _, _ = aux.load_data(data)

    # Determine animation bounds
    Z_bounds = get_animation_bounds(Z) if ZorC == 'Z' else box
    C_bounds = get_animation_bounds(C) if ZorC == 'C' else box

    # Initialize plot
    fig, ax = initialize_plot(Dim)

    # Create and save the animation
    ani = create_animation(fig, ax, Z, C, ZorC, Dim, Z_bounds, C_bounds, tf)
    save_animation(ani, ZorC, Dim)

def get_animation_bounds(frames):
    """
    Calculate the bounds for the animation based on the data frames.
    """
    all_points = np.concatenate(frames)
    min_bounds = np.min(all_points, axis=0)
    max_bounds = np.max(all_points, axis=0)
    return [min_bounds[0], min_bounds[1], min_bounds[2], max_bounds[0], max_bounds[1], max_bounds[2]]

def initialize_plot(Dim):
    """
    Initialize the plot based on the specified dimension.
    """
    fig = plt.figure()
    fig.set_size_inches(10, 10, True)
    if Dim == '2D':
        ax = fig.add_subplot()
    elif Dim == '3D':
        ax = fig.add_subplot(projection='3d')
    else:
        raise ValueError('Invalid dimension specified. Please choose "2D" or "3D".')
    return fig, ax

def create_animation(fig, ax, Z, C, ZorC, Dim, Z_bounds, C_bounds, tf):
    """
    Create the animation object.
    """
    Ndt = len(Z)
    update_func = get_update_function(ax, Z, C, ZorC, Dim, Z_bounds, C_bounds)
    return animation.FuncAnimation(fig, update_func, frames=Ndt, interval=tf)

def get_update_function(ax, Z, C, ZorC, Dim, Z_bounds, C_bounds):
    """
    Returns the appropriate update function for the animation.
    """
    def update(i):
        ax.cla()
        if ZorC == 'Z':
            plot_data(ax, Z[i], Dim, Z_bounds, ZorC)
        elif ZorC == 'C':
            plot_data(ax, C[i], Dim, C_bounds, ZorC)
        else:
            raise ValueError('Invalid ZorC value. Choose "Z" for seeds or "C" for centroids.')

    return update

def plot_data(ax, data, Dim, bounds, ZorC):
    """
    Plot the data on the given axis based on the dimension.
    """
    if Dim == '2D':
        ax.scatter(data[:,0], data[:,1], c=data[:,2], cmap='jet', edgecolor='none', s=8)
        ax.set_xlim([bounds[0], bounds[3]])
        ax.set_ylim([bounds[1], bounds[4]])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    elif Dim == '3D':
        color = 'blue' if ZorC == 'Z' else 'red'  # Set color based on ZorC
        ax.scatter(data[:,0], data[:,1], data[:,2], color=color, s=8)
        ax.set_xlim([bounds[0], bounds[3]])
        ax.set_ylim([bounds[1], bounds[4]])
        ax.set_zlim([bounds[2], bounds[5]])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

def save_animation(ani, ZorC, Dim):
    """
    Save the animation to a file.
    """
    filename = f'./animations/SG_{"Seeds" if ZorC == "Z" else "Centroids"}_{Dim}.gif'
    FFwriter = animation.FFMpegWriter(fps=30)
    ani.save(filename, writer=FFwriter, dpi=100)

# Example usage
# point_animator('data_file', 'Z', '2D', [0, 0, 0, 10, 10, 10], 1000)