import numpy as np
from FEM import FEM_space
################################################################################
savepath = "figures/"
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
# set global font sizes for all matplotlib text elements
plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 22,
})

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(x_fine.flatten(), fu_fine.flatten(), color="blue", linestyle = "-", \
         label="implemented flux function")
ax.plot(x_fine.flatten(), fu_exact_fine.flatten(), color="red", \
         linestyle = "--", label="exact flux function")
ax.grid()
ax.legend()
plt.savefig(savepath + "flux_function.png", dpi=300)