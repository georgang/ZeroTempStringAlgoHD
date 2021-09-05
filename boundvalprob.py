import numpy as np
import matplotlib.pyplot as plt
from Landscape import *
from scipy.integrate import solve_bvp
from matplotlib.ticker import LinearLocator


# Write 2nd order ODE as a system of 1st order ODE.
# t as (#of points,), x as (2*dim,#ofpoints), result as (2*dim,#ofpoints)
# TODO: Remove for-loop.
def func(t, x):
    result = np.zeros((x.shape[0],x.shape[1]))
    for i in range(t.size):
        result[0,i] = x[2,i]
        result[1,i] = x[3,i]
        result[2,i] = -E_grad(np.array([[x[0,i],x[1,i]]]))[0,0]
        result[3,i] = -E_grad(np.array([[x[0,i],x[1,i]]]))[0,1]
    return result


# Define boundary conditions.
# xa,xb as (2*dim,), result as (2*dim,)
def bc(xa, xb):
    return np.array([xa[0]-4, xa[1]-2, xb[0]-2, xb[1]-4])


if __name__ == '__main__':

    dim = 2
    # Define time interval.
    t = np.linspace(0, 1, 11)
    # Make initial guess for path.
    x_guess1 = np.zeros((2*dim, t.size))
    x_guess2 = np.vstack((np.linspace(10,-3,t.size)[np.newaxis,:],
                          np.linspace(-4,5,t.size)[np.newaxis,:],
                          np.linspace(-4,5,t.size)[np.newaxis,:],
                          np.linspace(-4,5,t.size)[np.newaxis,:]))

    # Solve boundary value problem.
    res = solve_bvp(func, bc, t, x_guess2, verbose=2)

    # Display the result.
    x_plot = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X = np.arange(np.min(gaussians[:, 0]) - 3, np.max(gaussians[:, 0]) + 3, 0.01)
    Y = np.arange(np.min(gaussians[:, 1]) - 3, np.max(gaussians[:, 1]) + 3, 0.01)
    X, Y = np.meshgrid(X, Y)
    Z = E(np.array([X.flatten(), Y.flatten()]).T)
    Z = np.reshape(Z, X.shape)
    surf = ax.plot_wireframe(X, Y, Z, color='black', linewidths=0.8)
    ax.set_zlim(np.min(Z), 10)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.02f}')
    ax.scatter(res.sol(x_plot)[0], res.sol(x_plot)[1],
               E(np.hstack((res.sol(x_plot)[0][:,np.newaxis],res.sol(x_plot)[1][:,np.newaxis]))),
               s=50, c='r', label='solution bvp')
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.legend(prop={'size': 15})
    plt.tight_layout()
    plt.show()