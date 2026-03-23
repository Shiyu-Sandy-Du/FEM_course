from RK import RK4_step
import numpy as np

################################################################################
savepath = "figures/"

################################################################################
# define du/dt = f(u, t) = cu
# note that t is a dummy argument in this test
def f(u, t, c=1.0):
    return c * u

# set up the constant for the ODE
c = 1.0

# set up the temporal discretisation
t_start = 0.0
t_end = 1.0
n_dt = 11
dt_largest = 0.5
dt_list = np.empty(n_dt)
dt_list[0] = dt_largest
u_end_list = np.empty(n_dt)
for i_dt in range(n_dt):
    if i_dt > 0:
        dt_list[i_dt] = dt_list[i_dt-1] / 2.0 
    dt = dt_list[i_dt]
    nt = int((t_end - t_start) / dt) + 1
    t = np.linspace(t_start, t_end, nt)

    # set up the initial condition
    u_0 = 1.0
    u = np.empty(nt)
    u[0] = u_0

    # perform time stepping
    for i_step in range(1, nt):
        u[i_step] = RK4_step(f, u[i_step - 1], t[i_step - 1], dt)
    u_end_list[i_dt] = u[-1]
# exact solution
u_exact_end = u_0 * np.exp(c*t_end)

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
dt_ref = np.linspace(0.1,0.001)
err_O4_ref = dt_ref ** 4
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(dt_list, np.abs(u_end_list - u_exact_end), color="red", marker="x", alpha=1.0, label="absolute error")
ax.plot(dt_ref, err_O4_ref, color="black", linestyle="--", alpha=0.5, label=r"$O(\Delta t ^4)$")
ax.set_xlabel(r"$\Delta t$")
ax.set_ylabel("error")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(dt_list[0],dt_list[-1])
ax.grid()
ax.legend()
plt.savefig(savepath + "RK4_test_convergence.png", dpi=300)
plt.close()