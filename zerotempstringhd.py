import time
import numpy as np
from Path import *
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator

if __name__ == '__main__':

    # Decide what should be illustrated.
    plotenergy = False
    plotaction = False

    # Import initial (a) and final coordinate (b) as (1,dim) each.
    a = np.array([np.loadtxt("InitialCoordinates2D.txt")[0,:]])
    b = np.array([np.loadtxt("InitialCoordinates2D.txt")[1,:]])

    # Create Path as (n_nodes,dim).
    path1 = Path(n_nodes=101, start=a, end=b)

    # Prepare illustrations.
    #energyaverage = np.array([])
    #energyintegral = np.array([])
    #action = np.array([])

    # Perform ZTSA.
    start = time.time()
    counter = 0
    while counter<10000 and np.amax(np.linalg.norm(path1.gradient_perp(),axis=1))>0.01:
        counter += 1
        #energyaverage = np.append(energyaverage, path1.energy_avg())
        #energyintegral = np.append(energyintegral, path1.energy_int())
        #action = np.append(action, path1.action())
        path1.evolve(stepsize=0.001, reparametrize=True, perpendicular=False)

    # Display time needed for convergence.
    print('The ZTSA took '+str(time.time()-start)+' seconds and '+str(counter)+' iterations to converge.')

    # Plot paths.
    #print(path1.nodes)

    # Check whether the resulting string is a solution of Erik's equation.
    # TODO: Make this more elegant and more dimensional.
    gradient_F = np.zeros(shape=(path1.n_nodes,2))
    for i in range(path1.n_nodes):
        gradient_F[i,0] = (F(np.array([[path1.nodes[i][0]+0.00001,path1.nodes[i][1]]]))
                            -F(path1.nodes[i][np.newaxis,:]))/0.00001
        gradient_F[i,1] = (F(np.array([[path1.nodes[i][0], path1.nodes[i][1]+0.00001]]))
                            -F(path1.nodes[i][np.newaxis,:]))/0.00001
    x_doubledot = path1.sec_der()

    if a.shape[1]==3:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.scatter(gaussians[:, 0],gaussians[:, 1],gaussians[:, 2], s=300, c='g', label='gaussian centers')
        ax.scatter(path1.nodes[:, 0], path1.nodes[:, 1], path1.nodes[:, 2], s=50, c='r', label='grad')
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.legend(prop={'size': 15})
        plt.tight_layout()
    if a.shape[1]==2:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        X = np.arange(np.min(gaussians[:,0])-3, np.max(gaussians[:,0])+3, 0.01)
        Y = np.arange(np.min(gaussians[:,1])-3, np.max(gaussians[:,1])+3, 0.01)
        X, Y = np.meshgrid(X, Y)
        Z = F(np.array([X.flatten(),Y.flatten()]).T)
        Z = np.reshape(Z, X.shape)
        surf = ax.plot_wireframe(X, Y, Z, color='black', linewidths=0.8)
        ax.set_zlim(np.min(Z), 1)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter('{x:.02f}')
        ax.scatter(path1.nodes[:, 0], path1.nodes[:, 1],
                   F(path1.nodes), s=50, c='r', label='final')
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.legend(prop={'size': 15})
        plt.tight_layout()

    #  Plot energy quantities as a function of the number of iterations.
    if plotenergy:
        fig_1, ax_1 = plt.subplots(1, 1)
        # Approximated energy integral.
        ax_1.plot(np.linspace(0, energyintegral.size, energyintegral.size), energyintegral, label='approximated energy path integral')
        # Average energy along the path.
        #ax_1.plot(np.linspace(0, energyaverage.size, energyaverage.size), energyaverage, label='average energy')
        ax_1.legend(prop={'size': 15})
        ax_1.set_ylabel("energy (a.u.)", size=20)
        ax_1.set_xlabel("number of iterations", size=20)
        ax_1.tick_params(axis='both', which='major', labelsize=20)
        plt.tight_layout()

    # Plot approximated action as a function of the number of iterations.
    if plotaction:
        fig_2, ax_2 = plt.subplots(1, 1)
        ax_2.plot(np.linspace(0, action.size, action.size), action, label='action')
        ax_2.legend(prop={'size': 15})
        ax_2.set_ylabel("action (a.u.)", size=20)
        ax_2.set_xlabel("number of iterations", size=20)
        ax_2.tick_params(axis='both', which='major', labelsize=20)
        plt.tight_layout()

    plt.show()