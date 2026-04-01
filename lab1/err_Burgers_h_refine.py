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
N_element_list = [8,16, 32, 64, 128, 256, 512]
# for N_element in N_element_list:
error_L2_list_collection = []
error_max_list_collection = []
error_H1_list_collection = []
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
    error_L2_list = []
    error_max_list = []
    error_H1_list = []

    for N_element in N_element_list:
        P = 7
        node_type = "GLL"

        domain = np.linspace(x_left, x_right, N_element + 1)
        Xh = FEM_space(node_type, P, domain, 1, "linear")

        ################################################################################
        x = Xh.mesh["x"]

        x0 = 0.0

        print("running case: " + case_name + "; h: " + str(np.round(domain[1]-domain[0], 3)))

        u0 = np.zeros_like(x)
        u0[x<x0] = uL
        u0[x>=x0] = uR


        BC_dirichlet = [uL,uR]
        RHS_f = RHS(nu2, Xh, BC_dirichlet)

        ################################################################################
        # set up the temporal discretisation
        t_start = 0.0
        t_end = 1.0
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
        ux_exact = -(uR-uL)**2/8.0/nu2 * sech2((x-s*t_end)*(uL-uR)/4.0/nu2)

        ################################################################################
        ux_current = RHS_f.Xh.conv(RHS_f.Xh.D, u_current) * RHS_f.Xh.mesh["Jinv"]
        error_L2 = np.sqrt(np.sum(((u_current - u_exact)**2)*Xh.mesh["B_sep"]))
        error_max = np.max(np.abs(u_current - u_exact))
        error_H1 = np.sqrt(np.sum(((u_current - u_exact)**2 + (ux_current - ux_exact)**2)*Xh.mesh["B_sep"]))
        error_L2_list.append(error_L2)
        error_max_list.append(error_max)
        error_H1_list.append(error_H1)
    error_L2_list_collection.append(np.array(error_L2_list))
    error_max_list_collection.append(np.array(error_max_list))
    error_H1_list_collection.append(np.array(error_H1_list))

import matplotlib.pyplot as plt
h = (x_right - x_left) / np.array(N_element_list)
plt.figure(figsize=(8,6))
plt.loglog(h, error_max_list_collection[0], color="red", marker="o", label=r"$L^\infty$ error, $\nu^2 = 1e-3$")
plt.loglog(h, error_max_list_collection[1], color="blue", marker="o", label=r"$L^\infty$ error, $\nu^2 = 5e-3$")
plt.loglog(h, error_L2_list_collection[0], color="red", marker="x", label=r"$L^2$ error, $\nu^2 = 1e-3$")
plt.loglog(h, error_L2_list_collection[1], color="blue", marker="x", label=r"$L^2$ error, $\nu^2 = 5e-3$")
plt.loglog(h, error_H1_list_collection[0], color="red", marker="s", label=r"$H^1$ error, $\nu^2 = 1e-3$")
plt.loglog(h, error_H1_list_collection[1], color="blue", marker="s", label=r"$H^1$ error, $\nu^2 = 5e-3$")
plt.xlabel(r"$h$")
plt.ylabel("error")
plt.xlim(h.max(), h.min())
plt.legend()
plt.grid()
plt.savefig(savepath + "error_vs_h.png", dpi=300)

# print_array = np.empty((len(N_element_list), 6))

# print_array[:,0] = error_max_list_collection[0]
# print_array[:-1,1] = np.log(error_max_list_collection[0][1:]/error_max_list_collection[0][:-1])/np.log(0.5)
# print_array[:,2] = error_L2_list_collection[0]
# print_array[:-1,3] = np.log(error_L2_list_collection[0][1:]/error_L2_list_collection[0][:-1])/np.log(0.5)
# print_array[:,4] = error_H1_list_collection[0]
# print_array[:-1,5] = np.log(error_H1_list_collection[0][1:]/error_H1_list_collection[0][:-1])/np.log(0.5)

# print(np.array2string(print_array, formatter={'float': '{:.3e}'.format}))


# print_array = np.empty((len(N_element_list), 6))

# print_array[:,0] = error_max_list_collection[1]
# print_array[:-1,1] = np.log(error_max_list_collection[1][1:]/error_max_list_collection[1][:-1])/np.log(0.5)
# print_array[:,2] = error_L2_list_collection[1]
# print_array[:-1,3] = np.log(error_L2_list_collection[1][1:]/error_L2_list_collection[1][:-1])/np.log(0.5)
# print_array[:,4] = error_H1_list_collection[1]
# print_array[:-1,5] = np.log(error_H1_list_collection[1][1:]/error_H1_list_collection[1][:-1])/np.log(0.5)

# print(np.array2string(print_array, formatter={'float': '{:.3e}'.format}))
