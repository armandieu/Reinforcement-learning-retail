import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_values(V):

    def get_Z(x, y, usable_ace):
        if (x,y,usable_ace) in V:
            return V[x,y,usable_ace]
        else:
            return 0

    def get_figure(usable_ace, ax):
        x_range = np.arange(11, 22)
        y_range = np.arange(1, 11)
        X, Y = np.meshgrid(x_range, y_range)
        
        Z = np.array([get_Z(x,y,usable_ace) for x,y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player\'s Current Sum')
        ax.set_ylabel('Dealer\'s Showing Card')
        ax.set_zlabel('State Value')
        ax.view_init(ax.elev, -120)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(211, projection='3d')
    ax.set_title('Usable Ace')
    get_figure(True, ax)
    ax = fig.add_subplot(212, projection='3d')
    ax.set_title('No Usable Ace')
    get_figure(False, ax)
    plt.show()

def plot_policy(policy, height, width, pipe_gap=1, args={}):

    def get_Z(x, y):
        if (x,y) in policy:
            return policy[x,y]
        
        else:
            return 0 # 

    def get_figure(ax):
        x_limit = width
        y_max = height
        x_range = np.arange(0, x_limit)
        y_range = np.arange(0, y_max)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.array([[get_Z(x,y) for x in x_range] for y in y_range])
        vmin = np.min(Z)  # Minimum value in Z
        vmax = np.max(Z)  # Maximum value in Z
        surf = ax.imshow(Z, cmap=plt.get_cmap('viridis', None), 
                         vmin=vmin, vmax=vmax, 
                         extent=[0-0.5, x_limit-0.5, -0-0.5, y_max-0.5])
        plt.xticks(x_range)
        ax.set_xticklabels(x_range, rotation=90)
        plt.yticks(y_range)
        plt.gca().invert_yaxis()
        ax.set_xlabel('Qty in Stock')
        ax.set_ylabel('Expiration time')
        ax.grid(color='w', linestyle='-', linewidth=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(surf, cax=cax)  # Use vmin and vmax for ticks
        cbar.set_ticks(np.arange(0, x_limit, 5))
        cbar.set_ticklabels(np.arange(0, x_limit, 5))
        
    fig = plt.figure(figsize=(16, 15))
    ax = fig.add_subplot(121)
    ax.set_title(args.get('title','Policy'))
    get_figure(ax)
    plt.show()