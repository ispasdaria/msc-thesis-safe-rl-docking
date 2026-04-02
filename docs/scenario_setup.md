### PROBLEM SETUP ###


## Modeled problem

The modeled control problem includes the closing phase and the final approach phase of close-proximity rendezvous. These phases are treated as two sequential autonomous control tasks, each associated with a dedicated agent.

![Docking Phases](docking_phases.png) (Luo, J. Zhang, and Tang, 2014)

Motivation: This design is adopted because the two phases impose different control objectives and safety requirements: the closing phase emphasizes convergence toward the target, operating at larger separations with larger admissible position and velocity errors, whereas the final approach phase requires precise and constrained motion in the vicinity of the docking target, where the admissible region is much smaller and the tolerances are tighter. Additionally, the reward in the closing phase emphasizes reducing distance and velocity, while in the final approach phase it emphasizes low closing speed, alignment, corridor keeping, and terminal accuracy. The safety constraints near the target are also much stricter than those farther away. Using a dedicated agent for each phase allows the control strategy to be specialized to the characteristics of that phase.

### Controller 1 — Corridor capture

In this phase, the chaser reduces the relative distance further and transfers from the previous near-target point, P2 (a couple of km away from the target), to a station-keeping point, P3 (hundreds of meters from the target). The purpose of this controller is to bring the chaser into the correct region for the final straight-line approach. Therefore, Controller 1 is required to:

- reduce range to the target and stabilize at or near the handover point: move the chaser from the initial close-range state, P2, to a terminal point or corridor-entry point, P3, located hundreds of meters away from the target
- reduce relative velocity: slow the relative motion down as the range decreases
- prepare the geometry for final approach: bring the chaser into the correct side or region so that it can enter the final straight-line approach corridor

### Controller 2 — Terminal docking translation

Starting from P3, the chaser approaches the target along a straight line as much as possible and must satisfy strict requirements on relative position, velocity, attitude, and angular rate at contact. The purpose of this controller is to guide the chaser through the docking corridor and achieve admissible terminal contact conditions. Therefore, Controller 2 is required to:

- enter the approach gate or corridor and track a straight-line approach: once at or near P3, move into the permitted line of approach and continue along a straight line
- keep lateral error small: stay centered with respect to the docking point and the line-of-sight (LOS) cone
- continue reducing approach speed: ensure that the closing speed becomes smaller as contact approaches
- arrive with admissible terminal conditions: satisfy the required constraints on relative position, velocity, attitude, and angular rate at contact with the docking point, defined as the center of the target docking interface


### Docking geometry 

The **docking axis** is defined as the centerline of the target docking port, the axis normal to the target docking interface plane and pointing outward from the target port. This axis is denoted by `e_d` and is assumed to coincide with one axis of the target-centered frame. The docking corridor is aligned with this axis, which therefore represents the nominal approach direction of the chaser. !!!!!!!!!!!!!!!!!! may need modification after axis convention 

The **target docking-port center**, expressed relative to the target center of mass, is defined as a fixed offset along the docking axis and is assumed to be located on the front face of the target body.

The **chaser docking-port center**, expressed relative to the chaser center of mass, is defined as a fixed offset along the chaser body `x`-axis and is assumed to be located on the front face of the chaser body.


#### Docking port geometry (Oestreich, Linares, and Gondhalekar, 2020)

The relative-motion frame is centered at the target center of mass. Let

- r ∈ R^3 be the position of the chaser center of mass relative to the target center of mass, expressed in the target-centered frame
- q be the chaser attitude relative to the target-centered frame
- R(q) ∈ R^(3×3) be the rotation matrix corresponding to q, mapping vectors from the chaser body frame to the target-centered frame
- r_c ∈ R^3 be the position of the chaser docking-port center relative to the chaser center of mass, expressed in the chaser body frame
- r_t ∈ R^3 be the position of the target docking-port center relative to the target center of mass, expressed in the target-centered frame

Then the relative position of the chaser docking port with respect to the target docking port is defined as

r_p = r + R(q) * r_c - r_t

Docking contact is achieved when the two docking ports coincide: r_p = 0

If v denotes the relative velocity of the chaser center of mass and ω its angular velocity, then the relative velocity of the chaser docking port with respect to the target docking port can be written as

v_p = v + ω × (R(q) * r_c)

for a stationary target frame. This way, terminal docking conditions can be imposed on r_p, v_p, attitude error, and angular velocity. 


## Reference frames 

Target-centered local frame set at the origin of the target center of mass 
Earth-centered inertial frame: used to show the absolute position of the spacecrafts 


## Assumptions

Assumption: One spacecraft is modelled as the active chaser, responsible for performing the approach and control actions.
Consequence: 
Limitation:

Assumption: The other spacecraft is modelled as a passive target, with no active control input during the docking phase.
Consequence: 
Limitation:

Assumption: Phasing part (earlier rendezvous maneuvers during which the large orbital differences between the chaser and target, such as relative distance, phase angle, and orbital plane differences, are corrected) and the homing part (chaser has already autonomously acquired the target orbit and reduced the relative velocity, reaching the station-keeping point P2, located a couple of kilometers away from the target) have already been completed successfully before the start of the modeled motion, therefore, the modeled motion begins from P2. (Luo, J. Zhang, and Tang, 2014)
Consequence: 
Limitation:

Assumption: The control task terminates at first contact between the chaser and the target (contact dynamics, latching mechanisms, and other capture-related processes are outside the scope of this control problem). 
Consequence: 
Limitation:

Assumption: The docking axis is assumed to coincide with the `x`-axis of the target-centered reference frame. !!!!!!!!!!!!!!!!!!! may need modification after axis convention 
Consequence: 
Limitation:

Assumption: The target docking-port center is assumed to be collinear with the target center of mass and to be located on the front face of the target body.
Consequence: 
Limitation:

Assumption: The chaser docking-port center is assumed to be collinear with the chaser center of mass and to be located on the front face of the chaser body.
Consequence: 
Limitation:


