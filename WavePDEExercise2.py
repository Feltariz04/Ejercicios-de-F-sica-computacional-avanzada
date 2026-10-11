import numpy as np
import matplotlib.pyplot as plt


def run_wave_first_rows(L, dx, dt, c, t_final, f_func, g_func):
    """
    Explicit finite-difference solver for 1D wave eq, just first rows
    returns y_prev and y_curr.
    """
    r = c * dt / dx
    Nx = int(round(L / dx))
    nodes = Nx + 1
    x = np.linspace(0, L, nodes)
    nt = int(round(t_final / dt))
    # initial conditions
    f = f_func(x)
    g = g_func(x)
    y_prev = f.copy()            # j = 0
    y_curr = np.zeros_like(x)    # j = 1
    # initialize second row (improved, second-order)
    ## this is the crucial line
    for i in range(1, nodes-1):
        y_curr[i] = f[i] + dt * g[i] + 0.5 * (r**2) * (f[i+1] + f[i-1] - 2*f[i])
    y_curr[0] = 0.0
    y_curr[-1] = 0.0
    ##
    return y_prev,y_curr

def run_wave(L, dx, dt, c, t_final, f_func, g_func):
    """
    Explicit finite-difference solver for 1D wave eq:
      y_tt = c^2 y_xx
    Returns dict of snapshots at requested times and final arrays.
    """

    r = c * dt / dx
    Nx = int(round(L / dx))
    nodes = Nx + 1
    x = np.linspace(0, L, nodes)
    nt = int(round(t_final / dt))
    
    y_prev,y_curr = run_wave_first_rows(L, dx, dt, c, t_final, f_func, g_func)
    # snapshots at these times (seconds)
    snapshot_times = [0.0, 0.125, 0.25, 0.375, 0.5]
    snapshot_steps = [int(round(t / dt)) for t in snapshot_times]
    
    snapshots = {0.0: y_prev.copy()}
    # time marching
    for j in range(1, nt+1):
        y_next = np.zeros_like(x)
        for i in range(1, nodes-1):
            y_next[i] = 2*y_curr[i] - y_prev[i] + (r**2) * (y_curr[i+1] + y_curr[i-1] - 2*y_curr[i])
        y_next[0] = 0.0
        y_next[-1] = 0.0
        # advance
        y_prev, y_curr = y_curr, y_next
        if j in snapshot_steps:
            snapshots[round(j*dt, 6)] = y_curr.copy()
    return {
        "x": x,
        "snapshots": snapshots,
        "r": r,
        "dt": dt,
        "dx": dx,
        "t_final": t_final
    }

# Problem definition
L = 1.0
dx = 0.01
dt_stable = 0.005    # stable run
dt_unstable = 0.01  # intentionally unstable (r > 1)
c = 1.0
t_final = 0.5
#New values
x0 = 0.5
sigma = 0.05

## define the functions that appear as boundary conditions
f_func = lambda x: np.exp(-((x-x0)**2)/2 * sigma)
g_func = lambda x: 0.2 * np.sin(np.pi * x)

# Run stable
result_stable = run_wave(L, dx, dt_stable, c, t_final, f_func, g_func)
print("Stable run: dx=", dx, "dt=", dt_stable, "r=", result_stable["r"])

# Run unstable
result_unstable = run_wave(L, dx, dt_unstable, c, t_final, f_func, g_func)
print("Unstable run: dx=", dx, "dt=", dt_unstable, "r=", result_unstable["r"])

# Plotting: stable vs unstable snapshots side-by-side
times = sorted(result_stable["snapshots"].keys())  # should be [0,0.25,0.5,0.75,1.0]
x = result_stable["x"]

plt.figure(figsize=(12, 8))
for k, t in enumerate(times):
    plt.subplot(3, 2, k+1)
    y_s = result_stable["snapshots"].get(t)
    y_u = result_unstable["snapshots"].get(t)
    # stable
    if y_s is not None:
        plt.plot(x, y_s, label=f"stable dt={dt_stable}, r={result_stable['r']:.3f}")
    # unstable (may blow up or be noisy)
    if y_u is not None:
        plt.plot(x, y_u, '--', label=f"unstable dt={dt_unstable}, r={result_unstable['r']:.3f}")
    plt.title(f"t = {t:.2f} s")
    plt.xlabel("x (m)")
    plt.ylabel("y (arb. units)")
    plt.grid(True)
    plt.legend()
# tidy layout
plt.tight_layout()
plt.show()
