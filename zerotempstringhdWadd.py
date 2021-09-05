import numpy as np
from PathWadd import *
import time
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Import initial (a) and final coordinate (b) as (1,dim) each.
    a = (np.loadtxt("beta_1e-1_unique_minima.txt")[:,0])[np.newaxis,:]
    b = (np.loadtxt("beta_1e-1_unique_minima.txt")[:,1])[np.newaxis,:]

    # Create Path as (n_nodes,dim).
    pathinitial = Path(n_nodes=101, start=a, end=b)
    path = Path(n_nodes=101, start=a, end=b)

    # Prepare illustrations.
    #energyaverage = np.array([])
    #energyintegral = np.array([])
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Perform ZTSA.
    start = time.time()
    counter = 0
    while counter<50:
        counter += 1
        #energyaverage = np.append(energyaverage, path1.energy_avg())
        #energyintegral = np.append(energyintegral, path1.energy_int())
        path.evolve(stepsize=0.0001, reparametrize=True, perpendicular=False)

    # Display time needed for convergence.
    print('The ZTSA took '+str(time.time()-start)+' seconds and '+str(counter)+' iterations to converge.')

    # Plot paths
    if a.shape[1]==3:
        ax.scatter(pathinitial.nodes[:, 0],pathinitial.nodes[:, 1],pathinitial.nodes[:, 2], s=50, c='y', label='initial path')
        ax.scatter(gaussians[:, 0],gaussians[:, 1],gaussians[:, 2], s=300, c='g', label='gaussian centers')
        ax.scatter(path.nodes[:, 0], path.nodes[:, 1], path.nodes[:, 2], s=50, c='r', label='grad')
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.legend(prop={'size': 15})
        plt.tight_layout()
    else:
        print(path.nodes)

    # Plot energy quantities as a function of the number of iterations.
    # fig_1, ax_1 = plt.subplots(1, 1)
    # # Approximated energy integral.
    # # # ax_1.plot(np.linspace(0, energyintegral.size, energyintegral.size), energyintegral, label='approximated energy path integral')
    # # Average energy along the path.
    # ax_1.plot(np.linspace(0, energyaverage.size, energyaverage.size), energyaverage, label='average energy')
    # ax_1.legend(prop={'size': 15})
    # ax_1.set_ylabel("energy (a.u.)", size=20)
    # ax_1.set_xlabel("number of iterations", size=20)
    # ax_1.tick_params(axis='both', which='major', labelsize=20)
    # plt.tight_layout()

    #plt.show()