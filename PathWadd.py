import numpy as np
from LandscapeWadd import *
from scipy.interpolate import interp1d

class Path:

    # n_nodes as int, start and end each as (1,dim)
    def __init__(self, n_nodes, start, end):
        self.n_nodes = n_nodes
        self.nodes = np.linspace(start[0], end[0], n_nodes)

    # Return cumulative arclength as (n_nodes,).
    def cum_arclength(self):
        return np.hstack([0, np.cumsum(np.linalg.norm(self.nodes[1:]-self.nodes[:-1], axis=1))])

    # Return differential arclength as (n_nodes-1,).
    def diff_arclength(self):
        return np.diff(self.cum_arclength())

    # Return approximated integrated energy.
    def energy_int(self):
        return np.dot(self.diff_arclength(),(E(self.nodes[1:])+E(self.nodes[:-1])/2))

    # Return integrated energy per distance.
    def energy_avg(self):
        return self.energy_int()/(self.cum_arclength()[self.n_nodes-1])

    # Return energy gradient as (n_nodes,dim) at every node.
    def gradient(self):
        return E_grad(self.nodes)

    # Return component of energy gradient perpendicular to path as (n_nodes-2,dim).
    def gradient_perp(self,central=True):
        if central:
            tau = (self.nodes[2:]-self.nodes[:-2])/(np.linalg.norm(self.nodes[2:]-self.nodes[:-2],axis=1)[:,np.newaxis])
            grdnt_perp = self.gradient()[1:-1]-np.sum(self.gradient()[1:-1]*tau,axis=1)[:,np.newaxis]*tau
        else:
            tau = (self.nodes[1:]-self.nodes[:-1])/(np.linalg.norm(self.nodes[1:]-self.nodes[:-1],axis=1)[:,np.newaxis])
            grdnt_perp = self.gradient()[1:-1]-np.sum(self.gradient()[1:-1]*tau[:-1],axis=1)[:,np.newaxis]*tau[:-1]
        return grdnt_perp

    # Evolve path according to gradient or gradient_perp and given stepsize.
    def evolve(self, stepsize, reparametrize=False, perpendicular=True):
        if perpendicular:
            self.nodes[1:-1] -= stepsize * self.gradient_perp()
        else:
            self.nodes[1:-1] -= stepsize * self.gradient()[1:-1]
        if reparametrize:
            arclength = self.cum_arclength()/(self.cum_arclength()[-1])
            self.nodes = interp1d(arclength,self.nodes,kind='cubic',axis=0)(np.linspace(0,1,self.n_nodes))