# This is a library for Runge-Kutta time stepping for an ODE y = dx/dt = f(x, t)

## Time stepping for 4th order RK scheme
# f: the rate of change dx/dt = f(x, t)
# x_current: the field on time stpe n
# dt: time step size
def RK4_step(f, x_current, t, dt):
    k1 = f(x_current, t)
    k2 = f(x_current + k1 * dt/2.0, t + dt/2.0)
    k3 = f(x_current + k2 * dt/2.0, t + dt/2.0)
    k4 = f(x_current + k3 * dt,     t + dt)
    return x_current + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)