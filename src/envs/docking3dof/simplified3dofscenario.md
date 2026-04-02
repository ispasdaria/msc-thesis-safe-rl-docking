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


## Assumptions

Assumption: The model takes into account the in-plane relative motion and the corresponding planar attitude motion. The out-of-plane translational motion and the two remaining rotational degrees of freedom are neglected.
Consequence: 
Limitation: