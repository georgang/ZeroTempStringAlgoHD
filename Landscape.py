import numpy as np

# Import centers of Gaussians as (#ofGaussians,dim), define dimension and parameter beta.
# TODO: Gaussians also as (1,dim) if there is only one.
gaussians = np.loadtxt("GaussianCenters2D.txt")
dim = gaussians.shape[1]
beta = 1.0

# Define energy landscape (Eq. 24 + normalisation). Coordinates as (#ofPoints,dim). Energy as (#ofPoints,).
def E(coordinates):
    nrgy = np.sum(np.exp(-(beta/2.0)*np.sum((coordinates[:,np.newaxis,:]-gaussians)**2,axis=2)),axis=1)
    nrgy = -(1.0/beta)*np.log((nrgy/gaussians.shape[0])*(beta/(2*np.pi))**(dim/2.0))
    return nrgy

# Calculate energy gradient as (#ofPoints,dim). Coordinates as (#ofPoints,dim).
def E_grad(coordinates):
    grdnt = np.exp(-(beta/2.0)*np.sum((coordinates[:,np.newaxis,:]-gaussians)**2,axis=2))[:,:,np.newaxis]
    grdnt = np.sum((coordinates[:,np.newaxis,:]-gaussians)*grdnt,axis=1)
    grdnt = grdnt/np.exp(-beta*E(coordinates)[:,np.newaxis])
    return (1/gaussians.shape[0])*grdnt*(beta/(2*np.pi))**(dim/2.0)

# Calculate F according to Erik's derivation. Coordinates as (#ofPoints,dim). F as (#ofPoints,).
def F(coordinates):
    return -(0.5)*np.sum(E_grad(coordinates)**2,axis=1)

# Approximate the gradient of F as (#ofPoints,dim). Coordinates as (#ofPoints,dim).
# TODO
def F_grad(coordinates):
    return F(coordinates)-F(coordinates)