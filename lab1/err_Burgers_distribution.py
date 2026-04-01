import numpy as np
from FEM import FEM_space
from RK import RK4_step
from RHS import RHS
def sech2(x):
    ax = np.abs(x)
    exp_ax = np.exp(-ax)
    return 4 * exp_ax**2 / (1 + exp_ax**2)**2
################################################################################
savepath = "figures/"
    
################################################################################
x_left = -4.0
x_right = 4.0
N_element = 40
P = 7
node_type = "GLL"

domain = np.linspace(x_left, x_right, N_element + 1)
Xh = FEM_space(node_type, P, domain, 1, "linear")

################################################################################
x = Xh.mesh["x"]

x0 = 0.0
err_max_list = []
err_L2_list = []
err_H1_list = []
for i_case in [0,3]:
    if i_case == 0:
        case_name = "burgers_shock"
        uL = 1.0
        uR = 0.0
        dt = 1e-4
        nu2 = 1e-3
    elif i_case == 1:
        case_name = "burgers_trivial"
        uL = 0.0
        uR = 0.0
        dt = 1e-2
        nu2 = 1e-3
    elif i_case == 2:
        case_name = "burgers_rarefaction"
        uL = 0.0
        uR = 1.0
        dt = 1e-2
        nu2 = 1e-3
    elif i_case == 3:
        case_name = "burgers_shock_strong_stab"
        uL = 1.0
        uR = 0.0
        dt = 1e-4
        nu2 = 5e-3
    print("running case: " + case_name)

    u0 = np.zeros_like(x)
    u0[x<x0] = uL
    u0[x>=x0] = uR


    BC_dirichlet = [uL,uR]
    RHS_f = RHS(nu2, Xh, BC_dirichlet)

    ################################################################################
    # set up the temporal discretisation
    t_start = 0.0
    t_end = 2.0
    nt = int((t_end - t_start) / dt) + 1
    t = np.linspace(t_start, t_end, nt)

    u_current = u0

    # perform time stepping
    for i_step in range(1, nt):
        # print("time step:", i_step, "current time:", t[i_step])
        u_current = RK4_step(RHS_f.RHS_maker, u_current, t[i_step - 1], dt)
        # implement the BC at the end of each time step (a RK step)
        u_current = RHS_f.dirichlet_BC_apply(u_current)

    ################################################################################
    # the exact solution
    s = (uL + uR)/2.0
    # s = 0.0
    u_exact = (uL + uR)/2.0 + (uR-uL)/2.0 * np.tanh((x-s*t_end)*(uL-uR)/4.0/nu2)

    ################################################################################
    # refine the results by spectral interpolation
    import gll_lib
    n_intp = 20
    xi, _ = gll_lib.gLLNodesAndWeights(P+1)
    x_hat = np.linspace(-1, 1, n_intp)
    phi = gll_lib.interp_matrix_1D(xi, x_hat)
    x_fine = x @ phi.T
    u_fine = u_current @ phi.T

    u_exact_fine = (uL + uR)/2.0 + (uR-uL)/2.0 * np.tanh((x_fine-s*t_end)*(uL-uR)/4.0/nu2)
    err = u_current - u_exact
    err_L2 = np.empty_like(u_current)
    err_H1 = np.empty_like(u_current)
    err_max = np.empty_like(u_current)
    for i_elem in range(Xh.nelem):
        err_L2[i_elem,:] = np.sqrt(np.sum(err[i_elem,:]**2 * Xh.mesh["B_sep"][i_elem,:]))
        err_H1[i_elem,:] = np.sqrt(np.sum((err[i_elem,:]**2 + (Xh.D @ err[i_elem,:])**2) * Xh.mesh["B_sep"][i_elem,:]))
        err_max[i_elem,:] = np.max(np.abs(err[i_elem,:]))
    err_L2_list.append(err_L2)
    err_H1_list.append(err_H1)
    err_max_list.append(err_max)

    # check for a heat equation
    # from scipy.special import erfc
    # u_exact_fine = 1.0/2.0 * erfc(x_fine/2/np.sqrt(nu2*t_end))

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
ax.plot(x.flatten(), err_max_list[0].flatten(), color="blue", linestyle = "-", \
        label=r"$\nu^2 = 1e-3$")
ax.plot(x.flatten(), err_max_list[1].flatten(), color="red", linestyle = "-", \
        label=r"$\nu^2 = 5e-3$")
ax.grid()
ax.legend()
plt.savefig(savepath + "err_distribution_max.png", dpi=300)

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(x.flatten(), err_L2_list[0].flatten(), color="blue", linestyle = "-", \
        label=r"$\nu^2 = 1e-3$")
ax.plot(x.flatten(), err_L2_list[1].flatten(), color="red", linestyle = "-", \
        label=r"$\nu^2 = 5e-3$")
ax.grid()
ax.legend()
plt.savefig(savepath + "err_distribution_L2.png", dpi=300)

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(x.flatten(), err_H1_list[0].flatten(), color="blue", linestyle = "-", \
        label=r"$\nu^2 = 1e-3$")
ax.plot(x.flatten(), err_H1_list[1].flatten(), color="red", linestyle = "-", \
        label=r"$\nu^2 = 5e-3$")
ax.grid()
ax.legend()
plt.savefig(savepath + "err_distribution_H1.png", dpi=300)