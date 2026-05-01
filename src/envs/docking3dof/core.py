import numpy as np 

# Frame convention: target-centered orbital frame with F_O = {V-bar, H-bar, R-bar}
# V-bar (x-axis) = along-track / approximately target velocity direction
# H-bar (y-axis) = orbit-normal direction
# R-bar (z-axis) = radial direction

# State:

# state[0:1] = relative position of chaser CoM w.r.t. target CoM in x and y directions in target-centered orbital frame 
# state[2:3] = relative velocity of chaser CoM w.r.t. target CoM in x and y directions in target-centered orbital frame 
# state[4] = relative attitude of chaser w.r.t target in target-centered orbital frame 
# state[5] = relative angular velocity of chaser w.r.t target in target-centered orbital frame 

state = np.array([rx, ry, vx, vy, psi, omegaz])


# Action (control input): 

# input[0:1] = commanded translational acceleration in x and y directions in target-centered orbital frame 
# input[2] = commanded angular acceleration about z-axis in target-centered orbital frame 

input = np.array([ax, ay, alphaz])

AX_MAX = 0.01   # m/s^2
AY_MAX = 0.01   # m/s^2
ALPHAZ_MAX = 0.003   # rad/s^2





