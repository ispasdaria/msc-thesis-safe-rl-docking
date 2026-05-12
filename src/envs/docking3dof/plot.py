import matplotlib.pyplot as plt

times, states, actions, rewards = env.get_history_arrays()

rx = states[:, 0]
ry = states[:, 1]

plt.figure()
plt.plot(rx, ry)
plt.xlabel("rx along V-bar [m]")
plt.ylabel("ry along H-bar [m]")
plt.title("3-DoF Relative Trajectory")
plt.grid(True)
plt.axis("equal")
plt.show()