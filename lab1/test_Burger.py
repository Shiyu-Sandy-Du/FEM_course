import numpy as np
from FEM import FEM_space

################################################################################
x_left = -5.0
x_right = 5.0
N_element = 8
P = 5
node_type = "GLL"

domain = np.linspace(x_left, x_right, N_element + 1)
Xh = FEM_space(node_type, P, domain, 1, "linear")

################################################################################
x = Xh.mesh["x"]
u = np.exp(-(x*x))
nu = 0.0
fu_weak = Xh.flux_avgC0(u, nu)
fu = fu_weak/Xh.mesh["B"]
fu_exact = 2*x*np.exp(-2*(x*x)) + nu*nu* (2 - 4*x*x) *np.exp(-(x*x))

################################################################################
# refine the results by spectral interpolation
import gll_lib
n_intp = 20
xi, _ = gll_lib.gLLNodesAndWeights(P+1)
x_hat = np.linspace(-1, 1, n_intp)
phi = gll_lib.interp_matrix_1D(xi, x_hat)
x_fine = x @ phi.T
fu_fine = fu @ phi.T
fu_exact_fine = fu_exact @ phi.T

################################################################################
# plot
import matplotlib.pyplot as plt
plt.figure()
plt.plot(x_fine.flatten(), fu_fine.flatten(), color="blue", linestyle = "-")
plt.plot(x_fine.flatten(), fu_exact_fine.flatten(), color="red", linestyle = "--")
plt.grid()
plt.show()