env = Docking3DoFEnv(dt=1.0, integrator="rk4", seed=1)

state = env.reset(
    initial_state=np.array([
        300.0,              # rx [m]
        5.0,                # ry [m]
        -0.03,              # vx [m/s]
        0.0,                # vy [m/s]
        np.deg2rad(2.0),    # psi [rad]
        0.0,                # omegaz [rad/s]
    ])
)

for k in range(1000):
    action = np.array([0.0, 0.0, 0.0])

    state, reward, done, info = env.step(action)

    if done:
        print("Episode ended at step:", k)
        print("Info:", info)
        break

print("Final state:", state)