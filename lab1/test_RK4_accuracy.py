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
dt = 0.01
nt = int((t_end - t_start) / dt) + 1
t = np.linspace(t_start, t_end, nt)

# set up the initial condition and allocate an array for the solution
u_0 = 1.0

u = np.empty(nt)
u[0] = u_0

# perform time stepping
for i_step in range(1, nt):
    u[i_step] = RK4_step(f, u[i_step - 1], t[i_step - 1], dt)

# exact solution
u_exact = u_0 * np.exp(c*t)

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
ax.plot(t, u, color="blue", linestyle="-", alpha=1.0, label="implemented RK4")
ax.plot(t, u_exact, color="red", linestyle="--", alpha=0.5, label="exact solution")
ax.set_xlabel("t")
ax.set_ylabel("u")
# ax.set_yscale("log")
ax.set_xlim(t[0],t[-1])
ax.grid()
ax.legend()
plt.savefig(savepath + "RK4_test_solution.png", dpi=300)
plt.close()

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(t, np.abs(u - u_exact)/u_exact, color="red", linestyle="--", label="relative error")
ax.set_xlabel("t")
ax.set_ylabel("error")
ax.set_xlim(t[0],t[-1])
ax.grid()
ax.legend()
plt.savefig(savepath + "RK4_test_error.png", dpi=300)
plt.close()