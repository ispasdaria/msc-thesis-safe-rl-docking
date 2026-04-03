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

The **docking axis** is defined as the centerline of the target docking port, the axis normal to the target docking interface plane and pointing outward from the target port. This axis is denoted by `e_d` and is assumed to coincide with the 'x'-axis of the target-centered frame. The docking corridor is aligned with this axis, which therefore represents the nominal approach direction of the chaser. 

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


#### Vehicles (International Space Station Multilateral Coordination Board, 2013)

The chaser is assumed to be a Boeing Starliner-like vehicle, and the target is assumed to be an ISS-like port module. The selected vehicles are approximated using the IDSS reference vehicle classes rather than exact proprietary spacecraft data. The Starliner-like chaser is approximated using th IDSS-15T reference class, since Starliner is a medium crew visiting vehcile and the IDSS defines medium vehicles in the range of 8 to 25 tonnes. The ISS-like target port module is approximated using the IDSS-350T reference class, since the IDSS states that these vehicles may dock to large space complexes in the range of 100 to 375 tonnes. 

Motivation: The Starliner is an IDSS/NDS-compatible visiting vehicle, while the ISS provides International Docking Adapter (IDA) interfaces for compatible spacecraft. This makes the scenatio realistic a standardized docking study, and keeps the geometric assumptions consistent with the IDSS framework from which the terminal contact conditions are used from. 

As a vehicle-to-vehicle docking case option, the analysis may also consider a Starliner-like chaser and a Crew Dragon-like target. Both the Starliner-like chaser and the Crew Dragon-like target are approximated using medium IDSS vehciles classes. The Starliner-like vehicles is approximated using the IDSS-15T class, and the Crew Dragon-like vehicles is approximated using the IDSS-10T class, as it is described by SpaceX to be capable of carrying up to seven people and listed as a spacecraft for Earth-orbit missions. 

Motivation: The study represents docking between two existing classes of IDSS-compatible crew vehicles of similar scale, which is useful for examining a more symmetric configuration. 

In the docking-port geometry model, the vectors from the vehicle center of mass to the docking-port center, r_c and r_t, are approximated using the IDSS reference mass-properties table. These values are given as the coordinates of the Hard Capture System mating plane center in a coordinate system located at the vehicle CoG, with the positive x-axis pointing toward the docking interface. These coordinates are used as representative first-order approximations of the CoM to docking-port offsets. 


## Reference frames (AE3202 Flight Dynamics Lecture notes)

- Earth-centered inertial frame: 

The Earth-centered inertial frame, denoted by F_I, is used to describe the absolute orbital motion of the target and chaser spacecraft. It is a right-handed orthogonal reference frame with origin at the Earth’s center of mass. The Z_I axis is aligned with the Earth’s rotation axis and points toward the North Pole, the X_I axis points toward the vernal equinox, and the Y_I axis completes the right-handed triad in the equatorial plane. This frame serves as the global reference frame for the absolute translational motion of both vehicles, so that their inertial positions and velocities can be defined in it. The attitude of each spacecraft may also be defined with respect to F_I, for example by Euler angles or quaternions. ![Earth-centered inertial reference frame](ECI.png)

- Target body-fixed frame/target-attached frame: 

The target body-fixed frame is a right-handed orthogonal frame fixed to the target vehicle, with origin at the target CoM. The x_{B_t} axis lies in the target symmetry plane and points forward toward the nominal docking interface, the z_{B_t} axis lies in the same plane and points downward, and the y_{B_t} axis points to the right, completing the right-handed triad. The target-centered frame F_T is chosen to coincide with the target body-fixed frame F_{B_t} (Oestreich, Linares, and Gondhalekar, 2020). The nominal docking axis is assumed to coincide with the positive x_T axis. Relative position, relative velocity, docking-port locations, corridor constraints, and terminal docking conditions are expressed in a frame attached to the target. The relative attitude of the chaser with respect to the target is described by the rotation from F_{B_c} to F_T. ![Body-Fixed Reference Frame](body_fixed_frame.png)

- Chaser body-fixed frame: 

The chaser body-fixed frame F_{B_c} is a right-handed orthogonal frame with origin at the chaser center of mass. The x_{B_c} axis lies in the symmetry plane and points forward, the z_{B_c} axis lies in the symmetry plane and points downward, and the y_{B_c} axis points to the right. This frame is used for quantities intrinsic to the chaser, such as thruster directions, the inertia matrix, body angular velocity, control forces and torques, and the docking-port offset relative to the chaser CoM. 

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

Assumption: The docking axis is assumed to coincide with the `x`-axis of the target-centered reference frame. 
Consequence: 
Limitation:

Assumption: The target docking-port center is assumed to be collinear with the target center of mass and to be located on the front face of the target body.
Consequence: 
Limitation:

Assumption: The chaser docking-port center is assumed to be collinear with the chaser center of mass and to be located on the front face of the chaser body.
Consequence: 
Limitation:

Assumption: The chaser is assumed to be a Boeing Starliner-like vehicle.
Consequence: 
Limitation:

Assumption: The Starliner-like chaser is approximated by the IDSS-15T reference vehicle class.
Consequence: 
Limitation:

Assumption: For the baseline docking scenario, the target is assumed to be an ISS-like target port module.
Consequence: 
Limitation:

Assumption: The ISS-like target port module is approximated by the IDSS-350T reference vehicle class.
Consequence: 
Limitation:

Assumption: For the alternative vehicle-to-vehicle docking scenario, the target is assumed to be a Crew Dragon-like vehicle.
Consequence: 
Limitation:

Assumption: The Crew Dragon-like target is approximated by the IDSS-10T reference vehicle class.
Consequence: 
Limitation:

Assumption: The vectors r_c and r_t from the vehicle center of mass to the docking-port center are approximated using the IDSS reference mass-properties table, which gives the coordinates of the Hard Capture System mating plane center in a coordinate system located at the vehicle center of gravity, with the positive x-axis directed toward the docking interface.
Consequence: 
Limitation:

Assumption: The target-centered frame F_T is assumed to coincide with the target body-fixed frame F_{B_t}.
Consequence: 
Limitation:

Assumption: 
Consequence: 
Limitation:

Assumption: 
Consequence: 
Limitation:

Assumption: 
Consequence: 
Limitation:


