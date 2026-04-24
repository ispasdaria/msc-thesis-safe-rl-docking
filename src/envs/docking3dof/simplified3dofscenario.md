### PROBLEM SETUP ###

## Modeled problem

The modeled control problem includes the closing phase and the final approach phase of close-proximity rendezvous. These phases are treated as two sequential autonomous control tasks, each associated with a dedicated agent.

### Controller 1 — Corridor capture

In this phase, the chaser reduces the relative distance further and transfers from the previous near-target point, P2, to a station-keeping or corridor-entry point, P3, located hundreds of meters from the target. In the simplified 3-DoF planar model, the purpose of this controller is to bring the chaser into the correct in-plane approach region for the final docking phase. Therefore, Controller 1 is required to:

- reduce the relative range and stabilize near the handover point: move the chaser from the initial close-range state, P2, to a terminal point or corridor-entry point, P3
- reduce the planar relative velocity: slow the in-plane relative motion as the chaser approaches P3
- prepare the planar approach geometry: bring the chaser to the correct side of the target and near the nominal in-plane docking axis so that the final approach can begin from an admissible corridor-entry condition
- prepare rotational alignment for terminal approach: reduce the relative attitude error and angular rate to values compatible with the start of the docking approach

### Controller 2 — Terminal planar docking

Starting from P3, the chaser approaches the target along the nominal in-plane docking axis and must satisfy strict terminal requirements at contact. In the simplified 3-DoF planar model, the purpose of this controller is to guide the chaser through the planar docking corridor and achieve admissible translational and rotational terminal conditions at the docking point. Therefore, Controller 2 is required to:

- enter and follow the planar docking corridor: once at or near P3, move into the permitted in-plane approach region and continue the approach approximately along the nominal docking axis
- keep the lateral deviation small: remain close to the nominal approach line and within the allowed planar corridor bounds
- continue reducing the closing speed: ensure that the relative approach speed decreases as contact is approached
- achieve admissible planar attitude alignment: reduce the relative attitude error and angular rate so that the docking interfaces are properly aligned in the simplified planar sense
- arrive with admissible terminal conditions at contact: satisfy the required constraints on relative position, planar relative velocity, relative attitude angle, and relative angular rate at the docking point, defined as the center of the target docking interface

### 3-DoF modeling interpretation

In this simplified model, the relative motion of the chaser with respect to the target is described in a planar setting with two translational degrees of freedom and one rotational degree of freedom. The state therefore captures the relative position of the chaser in the plane and its relative orientation with respect to the target docking frame. Out-of-plane motion, full three-dimensional attitude dynamics, angular rates about the other axes, and post-contact mechanical docking effects are not modeled. The control objective is to guide the chaser safely from the initial close-range region to first contact while satisfying admissible terminal conditions on both relative position and relative orientation.

Motivation: It provides a better representation of the docking task than a purely translational model, while still remaining manageable for analysis and implementation. In docking, it is not sufficient to bring the chaser close to the target; the vehicle must also reach an appropriate orientation for contact. Including an in-plane rotational state, it makes possible to account for interface alignment in addition to relative position and velocity. The model avoids the full complexity of a complete 6-DoF description, making it more suitable for early controller development and reinforcement learning experiments. It can be seen as a practical compromise between realism and simplicity, as well as a useful intermediate step toward a higher-fidelity model.

The relative motion is restricted to the x_T-y_T plane of the target-attached frame (Sharma et al., 2024). The translational DoFs are the relative displacements along the nominal docking axis x_T and the lateral axis y_T, while the rotational DoF is the relatieve in-plane attitude angle represented by a rotation about the out-of-plane z_T axis. 

Motivation: This choice allows the model to capture approach motion, lateral corridor deviation, and yaw-like docking-interface misalignment, while neglecting vertical motion and the remaining attitude dynamics. 

#### Docking-port geometry for the simplified planar model

The relative-motion frame is centered at the target center of mass. In the simplified planar model, the motion is restricted to a plane and described by two translational coordinates and one relative rotation angle. Let

- r = [x, y]^T be the position of the chaser center of mass relative to the target center of mass, expressed in the target-centered planar frame
- theta be the relative in-plane orientation of the chaser with respect to the target frame
- R(theta) be the planar rotation matrix corresponding to theta, mapping vectors from the chaser body frame to the target-centered planar frame
- r_c = [r_cx, r_cy]^T be the position of the chaser docking-port center relative to the chaser center of mass, expressed in the chaser body frame
- r_t = [r_tx, r_ty]^T be the position of the target docking-port center relative to the target center of mass, expressed in the target-centered planar frame

The relative position of the chaser docking port with respect to the target docking port is then defined as

r_p = r + R(theta) * r_c - r_t

where

R(theta) = [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]

Docking contact is achieved when the two docking-port centers coincide: r_p = 0

If v = [x_dot, y_dot]^T denotes the relative velocity of the chaser center of mass and omega = theta_dot the relative angular velocity about the out-of-plane axis, then the relative velocity of the chaser docking port with respect to the target docking port can be written as

v_p = v + omega * S * R(theta) * r_c

where

S = [[0, -1], [1, 0]]

is the planar skew-symmetric matrix. This term represents the additional docking-port velocity induced by the in-plane rotation of the chaser when the docking port is offset from the center of mass.

This formulation makes it possible to define terminal docking conditions at the docking-port level rather than at the center-of-mass level. In particular, docking success can be imposed through admissible bounds on the docking-port position error r_p, the docking-port relative velocity v_p, the relative orientation error theta, and the relative angular velocity omega.


## 3DoF Docking Problem Setupp 

### State 
x(t) = [r(t), v(t), theta(t), omega(t)]

where: 
- r(t) ∈ R^2: relative position in target plane (x_T, y_T)
- v(t) ∈ R^2: relative velocity in plane
- theta(t): relative in-plane orientation (rotation about z_T)
- omega(t): scalar angular velocity about z_T


### Control input 
u(t) = [f_c(t), tau_c(t)]

where: 
- f_c(t) ∈ R^2: planar control force
- tau_c(t): scalar torque about z_T 

### Admissible control set 
U = { u :
    ||f_c|| <= f_max AND
    |tau_c| <= tau_max
}


### Docking port relative position 
r_p = r + R(theta) * r_c - r_t

where: 
- r_c ∈ R^2: chaser docking port offset
- r_t ∈ R^2: target docking port offset
- R(theta): 2×2 rotation matrix
- r_p ∈ R^2

### Docking axis 
e_d = [1, 0]

#### Longitudinal distance 
s = dot(e_d, r_p)

#### Lateral offset 
rho = | y-component of r_p |


### Primitive Geometric Sets 

#### Approach Ellipsoid (ellipse now)
E_AE = { r :
    r^T Q_AE r <= 1
}

Q_AE = diag(1/2000^2, 1/1000^2)

#### Keep-Out Sphere (circle now)
S_KOS = { r :
    ||r|| <= 200
}

#### Docking Cone (2D wedge) 
C_dock = { x :
    0 <= s <= 2000 AND
    rho <= (200 / 2000) * s
}

#### Docking Corridor (strip)
C_corr = { x :
    0 <= s <= d_P3 AND
    rho <= 5
}


### Alignment Sets 

#### Safe alignment set 
A_safe = {
    |theta_e| <= 3 deg AND
    |omega| <= 0.15 deg/s
}

#### Warning alignment set 
A_warn = {
    (3 < |theta_e| <= 4) OR
    (0.15 < |omega| <= 0.2)
}

#### Unsafe alignment set 
A_unsafe = {
    |theta_e| > 4 OR
    |omega| > 0.2
}

### Velocity Sets 

#### Closing phase 
V_safe_close = {
    ||v_p|| <= 0.25 AND
    |s_dot| <= 0.15
}

V_warn_close = {
    (0.25 < ||v_p|| <= 0.35) OR
    (0.15 < |s_dot| <= 0.25)
}

V_unsafe_close = {
    ||v_p|| > 0.35 OR
    |s_dot| > 0.25
}

#### Final approach phase 
V_safe_final = {
    ||v_p|| <= 0.15 AND
    |s_dot| <= 0.1
}

V_warn_final = {
    (0.15 < ||v_p|| <= 0.2) OR
    (0.1 < |s_dot| <= 0.15)
}

V_unsafe_final = {
    ||v_p|| > 0.2 OR
    |s_dot| > 0.15
}


### Terminal Docking Set 
X_dock = {
    ||r_p|| <= 0.1 AND
    ||v_p|| <= 0.1 AND
    |theta_e| <= 4 deg AND
    |omega| <= 0.2 deg/s
}


### Initialization Sets 

#### P2 
X_P2 = {
    s ∈ [d_P2_min, d_P2_max],
    rho <= rho_P2,
    v_parallel ∈ [v_min_P2, v_max_P2],
    |v_perp| <= v_perp_P2,
    |theta_e| <= theta_P2,
    |omega| <= w_P2
}

#### P3 
X_P3 = {
    s ∈ [d_P3_min, d_P3_max],
    rho <= rho_P3,
    v_parallel ∈ [v_min_P3, v_max_P3],
    |v_perp| <= v_perp_P3,
    |theta_e| <= theta_P3,
    |omega| <= w_P3
}


### Safety Sets 

#### Closing phase 

#### Geometry layer 

##### Safe set 
G_safe_close = (E_AE \ S_KOS) ∩ C_dock   with   d_P3 <= s <= d_P2 
(region inside the approach ellipsoid but outside the keep-out sphere, and constrained to the docking cone between P2 and P3)

##### Warning set 
G_warn_close = (E_AE \ (S_KOS ∪ C_dock)) ∪ (S_KOS ∩ C_dock) 
(union of regions inside approach ellipsoid but outside the docking cone and keep-out sphere + regions inside the keep-out sphere but inside the docking cone --> geometrically risk but recoverable situations)

##### Unsafe set 
G_unsafe_close = S_KOS \ C_dock
(inside keep-out sphere but outside docking cone)

##### Outside set 
G_out_close = E_AE_complement

##### Combined sets
X_safe_close = G_safe_close ∩ V_safe_close

X_warn_close = (G_warn_close ∪ V_warn_close) \ X_unsafe_close

X_unsafe_close = G_unsafe_close ∪ V_unsafe_close

#### Final approach 

##### Geometry layer 

###### Safe set 
G_safe_final = C_dock ∩ C_corr
(chaser is inside docking cone and inside the final approach corridor)

###### Warning set 
G_warn_final = C_dock \ C_corr
(chaser is inside docking cone, but outisde stricter final corridor)

###### Unsafe set 
G_unsafe_final = C_dock_complement with   0 <= s <= d_P3
(chaser is outside docking cone)

X_safe_final = G_safe_final ∩ V_safe_final ∩ A_safe

X_warn_final = (G_warn_final ∪ V_warn_final ∪ A_warn) \ X_unsafe_final

X_unsafe_final = G_unsafe_final ∪ V_unsafe_final ∪ A_unsafe


### Phase Definitions 

#### Closing (P2 --> P3)
Initial:
x(t_P2) ∈ X_P2

During:
x(t) ∈ X_safe_close
u(t) ∈ U

Final:
x(t_P3) ∈ X_P3

#### Final Approach (P3 --> Docking)
Initial:
x(t_P3) ∈ X_P3

During:
x(t) ∈ X_safe_final
u(t) ∈ U

Final:
x(t_f) ∈ X_dock


## Assumptions

Assumption: The model takes into account the in-plane relative motion and the corresponding planar attitude motion. The out-of-plane translational motion and the two remaining rotational degrees of freedom are neglected.
Consequence: 
Limitation:

Assumption: The docking motion is assumed to be planar in the target-attached x_T-y_T plane, with relative translation along x_T and y_T, and relative in-plane rotation represented by an angle about z_T. 
Consequence: 
Limitation:

Assumption: In the planar 3-DoF scenario, the out-of-plane position and out-of-plane velocity are assumed to be zero.
Consequence: 
Limitation: