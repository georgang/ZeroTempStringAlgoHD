import numpy as np

gaussians = np.transpose(np.loadtxt("Ycell_inf"))
print(gaussians.shape)
print(0.1/(2.0*np.pi))