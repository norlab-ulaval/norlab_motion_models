import numpy as np
import matplotlib.pyplot as plt

# ---- Simulation parameters ----
T = 6.0        # total simulation time [s]
dt = 0.01       # time step [s]
t = np.arange(0, T + dt, dt)

# ---- Twist sets ----
lin_speeds = [0.5]       # m/s
ang_speeds = [0.0, 0.2, 0.4, 0.6, 1.0, 1.2,1.5,1.6,1.8, 2.0]  # rad/s

# ---- Plot ----
plt.figure(figsize=(8, 8))

for v_lin in lin_speeds:
    for v_ang in ang_speeds:

        # State initialization
        x = np.zeros_like(t)
        y = np.zeros_like(t)
        theta = np.zeros_like(t)

        # Initial pose
        x[0], y[0], theta[0] = 0.0, 0.0, 1.57

        # Simulation loop
        for i in range(1, len(t)):
            x[i] = x[i-1] + v_lin * np.cos(theta[i-1]) * dt
            y[i] = y[i-1] + v_lin * np.sin(theta[i-1]) * dt
            theta[i] = theta[i-1] + v_ang * dt

        # Label for legend
        label = f"v={v_lin}, ω={v_ang}"

        # Plot trajectory
        plt.plot(x, y, label=label)

# ---- Plot styling ----
plt.axis("equal")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title(f"Trajectories for Multiple Linear & Angular Speeds ({T} s)")
plt.grid(True)
plt.legend(fontsize=8, loc="best")
plt.show()
