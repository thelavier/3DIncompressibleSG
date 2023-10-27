import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#Animate the solution to the ODE
def point_animator(data, ZorC, Dim, box, tf):
    """
    Function animating the data produced by the optimal transport solver.

    Inputs:
        data: The data stored by the solver, must be a string
        ZorC: Decide if you want to animate the seeds or the weights, must also be a string
        Dim: Decide if you want to animate the seeds in 2D or 3D, must be a string
        box: list or tuple defining domain [xmin, ymin, zmin, xmax, ymax, zmax]
        tf: The 'Final time' for the solver, used to ensure that the frames and the animation interval are not jarring

    Outputs:
        animation: An animation of the seeds or the centroids depending on user choice
    """
    # Set up the animation 
    plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg'
    global Z
    global C

    # Load the data
    loaded_data = np.load(data)

    # Access the individual arrays
    Z = loaded_data['data1']
    C = loaded_data['data2']

    # Find the max and min of the seeds so that the animation domains are appropriately sized
    Zxmax = float('-inf')
    Zxmin = float('inf')
    Zymax = float('-inf')
    Zymin = float('inf')
    Zzmax = float('-inf')
    Zzmin = float('inf')

    for frame in Z:
        # Find min and maxs in the frame
        Zxmin_in_frame = np.min(frame[:, 0])
        Zxmax_in_frame = np.max(frame[:, 0])
        Zymin_in_frame = np.min(frame[:, 1])
        Zymax_in_frame = np.max(frame[:, 1])
        Zzmin_in_frame = np.min(frame[:, 2])
        Zzmax_in_frame = np.max(frame[:, 2])

        # Update the min and max values 
        Zxmin = min(Zxmin, Zxmin_in_frame)
        Zxmax = max(Zxmax, Zxmax_in_frame)
        Zymin = min(Zymin, Zymin_in_frame)
        Zymax = max(Zymax, Zymax_in_frame)
        Zzmin = min(Zzmin, Zzmin_in_frame)
        Zzmax = max(Zzmax, Zzmax_in_frame)

    # Establish Animation parameters
    Ndt = len(Z)

    # Create the plot
    fig = plt.figure()
    fig.set_size_inches(10, 10, True)
    if Dim == '2D':
        ax = fig.add_subplot()
    elif Dim == '3D':
        ax = fig.add_subplot(projection='3d')
    else:
        print('Please specify the dimension of the animation!')

    def update(i):
        global Z
        global C

        # Update the plot
        if ZorC == 'Z':
            if Dim == '2D':
                ax.cla()
                ax.scatter(Z[i][:,0], Z[i][:,1], c = Z[i][:,2], cmap = 'jet', edgecolor = 'none', s = 8)
                ax.set_xlim([Zxmin, Zxmax])
                ax.set_ylim([Zymin, Zymax])
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
            elif Dim == '3D':
                ax.cla()
                ax.scatter(Z[i][:,0], Z[i][:,1], Z[i][:,2], c = Z[i][:,2], cmap = 'jet', edgecolor = 'none', s = 8)
                ax.set_xlim([Zxmin, Zxmax])
                ax.set_ylim([Zymin, Zymax])
                ax.set_zlim([Zzmin, Zzmax])
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
            else:
                print('Please specify the dimension of the animation!')
        elif ZorC == 'C':
            if Dim == '2D':
                ax.cla()
                ax.scatter(C[i][:,0], C[i][:,1], c = Z[i][:,2], cmap = 'jet', edgecolor = 'none', s = 8)
                ax.set_xlim([box[0], box[3]])
                ax.set_ylim([box[1], box[4]])
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
            elif Dim == '3D':
                ax.cla()
                ax.scatter(C[i][:,0], C[i][:,1], C[i][:,2], c = Z[i][:,2], cmap = 'jet', edgecolor = 'none', s = 8)
                ax.set_xlim([box[0], box[3]])
                ax.set_ylim([box[1], box[4]])
                ax.set_zlim([box[2], box[5]])
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
            else:
                print('Please specify the dimension of the animation!')
        else:
            print('Please specify if you want to animate the centroids or the seeds!')

    if ZorC == 'Z':
        if Dim == '2D':
            ani = animation.FuncAnimation(fig, update, frames = Ndt, interval = tf)
            FFwriter = animation.FFMpegWriter(fps = 1000)
            ani.save('./animations/SG_Seeds_2D.gif', writer = FFwriter, dpi = 100)
        elif Dim == '3D':
            ani = animation.FuncAnimation(fig, update, frames = Ndt, interval = tf)
            FFwriter = animation.FFMpegWriter(fps = 1000)
            ani.save('./animations/SG_Seeds_3D.gif', writer = FFwriter, dpi = 100)
        else:
            print('Please specify the dimension of the animation!')
    elif ZorC == 'C':
        if Dim == '2D':
            ani = animation.FuncAnimation(fig, update, frames = Ndt, interval = tf)
            FFwriter = animation.FFMpegWriter(fps = 1000)
            ani.save('./animations/SG_Centroids_2D.gif', writer = FFwriter, dpi = 100)
        elif Dim == '3D':
            ani = animation.FuncAnimation(fig, update, frames = Ndt, interval = tf)
            FFwriter = animation.FFMpegWriter(fps = 1000)
            ani.save('./animations/SG_Centroids_3D.gif', writer = FFwriter, dpi = 100)
        else:
            print('Please specify the dimension of the animation!')
    else:
            print('Please specify if you want to animate the centroids or the seeds!')