import numpy as np
from Landscape import *
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

    # Return second derivative as (n_nodes,). delta_t = T/#ofnodes.
    # TODO: The second derivative depends on T.
    def sec_der(self,delta_t=20.0):
        forward = (self.nodes[2]-2*self.nodes[1]+self.nodes[0])/(delta_t)**2
        central = np.append(forward[np.newaxis,:],
                            (self.nodes[2:]-2*self.nodes[1:-1]+self.nodes[:-2])/(delta_t)**2, axis=0)
        return np.append(central, ((self.nodes[self.n_nodes-1] -
                                    2*self.nodes[self.n_nodes-2] +
                                    self.nodes[self.n_nodes-3])/(delta_t)**2)[np.newaxis,:], axis=0)

    # Return approximated integrated energy.
    def energy_int(self):
        return np.dot(self.diff_arclength(),(E(self.nodes[1:])+E(self.nodes[:-1])/2))

    # Return integrated energy per distance.
    def energy_avg(self):
        return self.energy_int()/(self.cum_arclength()[self.n_nodes-1])

    # Return action S_T[z(t)] as in Eq. 9 of Cameron_2011.
    def action(self,T=10):
        delta_t = T/self.n_nodes
        action = np.sum(((self.nodes[1:]-self.nodes[:-1])/delta_t
                            +self.gradient()[:-1])**2,axis=1)*delta_t
        return np.sum(action)

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