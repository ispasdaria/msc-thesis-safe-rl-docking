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


## Target Orbit and Relative Docking Geometry

### Target orbit

The target is assumed to move on an ISS-like circular low-Earth orbit, compatible with standard close-proximity relative-motion modeling assumptions. The target orbit is defined by the following parameters:

- orbit type: circular low-Earth orbit
- altitude: 400 km
- inclination: 51.6 deg
- eccentricity: 0
- semi-major axis: a = R_E + 400 km
- right ascension of the ascending node: 0 deg
- argument of perigee: 0 deg
- true anomaly at the initial time: 0 deg

Motivation: An ISS-like orbit is adopted because the baseline target is modeled as an ISS-like docking port module. A circular orbit simplifies the relative-motion modeling and is consistent with common assumptions used in close-proximity rendezvous and docking studies. The altitude of 400 km and inclination of 51.6 deg are chosen as representative values for an ISS-like low-Earth orbit. The eccentricity is set to zero in order to avoid additional complexity due to orbital ellipticity and to keep the relative-motion model consistent with a circular-reference-orbit formulation. The right ascension of the ascending node, argument of perigee, and initial true anomaly are fixed to zero for convenience, since the docking-control problem is formulated in the target-centered frame and these angles are not expected to significantly affect the local close-proximity docking scenario considered here.

### Reference docking geometry

The docking axis is assumed to coincide with the positive x_T-axis of the target-centered frame, and the chaser is initialized on the positive x_T side of the target and approaches the docking interface by moving toward smaller x_T values.

The target attitude is assumed to remain fixed during the modeled docking phase, and the target angular velocity is assumed to be zero. Therefore, the relevant rotational state is the attitude and angular velocity of the chaser relative to the target-centered frame.

Motivation: Aligning the docking axis with the positive x_T-axis provides a clear geometric interpretation of the approach direction and simplifies the definition of corridor constraints and terminal docking conditions. Initializing the chaser on the positive x_T side ensures that the vehicle approaches the target from the nominal docking direction. Assuming a fixed target attitude and zero target angular velocity is consistent with modeling the target as passive during docking and reduces the complexity of the relative rotational dynamics. Under these assumptions, the control problem can focus on the chaser relative to the target.

### Relative initial conditions and handover geometry

The modeled motion begins at the close-range rendezvous point P2, after phasing and homing have already been completed. At this point, the chaser is assumed to be in a controlled approach state: it is not yet in terminal docking, but it is already roughly aligned with the target and has relatively small translational and rotational errors.

The point P3 defines the end of the closing phase and the start of the terminal docking phase. It is selected as a corridor-entry or handover region from which the docking-port relative state is already sufficiently aligned for precise final approach and corridor tracking.

Motivation: P2 must therefore represent a realistic close-range state in which the chaser is already under controlled motion and roughly prepared for final approach, but is not yet sufficiently aligned to begin terminal docking. P3 is introduced as a handover state between the two controllers so that the closing controller and the terminal docking controller can address different objectives and operate under different tolerance levels. The conditions at P3 are chosen to be significantly tighter than those at P2, reflecting the fact that the chaser should already be better aligned, slower, and closer to the corridor before the terminal docking controller takes over.

## Nominal states and randomized initialization

To define representative but non-identical docking scenarios, nominal states are selected for both `P2` and `P3`, and the actual initial conditions are sampled from bounded regions around these nominal values.

Two sets of bounds are defined:

- a narrower testing range, used to evaluate controller performance under representative operating conditions
- a wider training range, used during learning to improve robustness to initial-state variations

Motivation: Using nominal states provides a clear reference scenario around which the docking problem can be defined. A narrower testing range ensures that the controller is evaluated on a realistic and focused set of operating conditions. A wider training range exposes the controller to a broader variety of initial conditions, which can improve robustness and reduce overfitting to a single nominal case. This approach is consistent with the idea of training on a broader region of the state space while evaluating performance on a more representative subset of that region (Oestreich, Linares, and Gondhalekar, 2020). 

### P2 definition

At P2, the chaser should therefore be reasonably well prepared for approach, while still exhibiting enough translational and rotational error to justify a dedicated closing controller.

#### Nominal P2 state

- axial distance along the docking axis: `1700 m`
- lateral offset: `0 m`
- out-of-plane offset:
  - `0 m` for the 3-DoF scenario
  - `0 m` nominal for the 6-DoF scenario
- axial relative velocity: `-0.10 m/s`
- lateral relative velocity: `0 m/s`
- out-of-plane relative velocity:
  - `0 m/s` for the 3-DoF scenario
  - `0 m/s` nominal for the 6-DoF scenario
- attitude error: `0 deg` on each axis
- angular velocity: `0 deg/s` on each axis

Motivation: A nominal axial distance of 1700 m is selected to place the chaser clearly within the close-range rendezvous region while remaining outside the terminal docking region. Zero nominal lateral and out-of-plane offsets represent an ideal reference alignment, around which perturbations can be introduced. A small nominal closing velocity is selected to reflect that the chaser is still approaching the target, but under controlled motion. Zero nominal attitude and angular-rate error represent the ideal reference handover geometry from the preceding homing phase.

#### P2 testing range

- axial distance: `1600 m` to `1800 m`
- lateral offset: `-100 m` to `100 m`
- out-of-plane offset:
  - `0 m` for the 3-DoF scenario
  - `-100 m` to `100 m` for the 6-DoF scenario
- axial relative velocity: `-0.15 m/s` to `-0.05 m/s`
- lateral relative velocity: `-0.02 m/s` to `0.02 m/s`
- out-of-plane relative velocity:
  - `0 m/s` for the 3-DoF scenario
  - `-0.02 m/s` to `0.02 m/s` for the 6-DoF scenario
- attitude error: `-3 deg` to `3 deg` on each axis
- angular velocity: `-0.05 deg/s` to `0.05 deg/s` on each axis

Motivation: The testing range is kept relatively narrow so that it represents a realistic close-range operating region after homing has already been completed. The chaser is allowed moderate position and orientation errors, but not large deviations that would be inconsistent with the assumption that the vehicle has already reached P2 under controlled motion. The lateral, out-of-plane, and angular errors are chosen to be noticeable but still small enough to remain representative of a pre-terminal docking state.

#### P2 training range

- axial distance: `1500 m` to `1900 m`
- lateral offset: `-150 m` to `150 m`
- out-of-plane offset:
  - `0 m` for the 3-DoF scenario
  - `-150 m` to `150 m` for the 6-DoF scenario
- axial relative velocity: `-0.20 m/s` to `-0.03 m/s`
- lateral relative velocity: `-0.03 m/s` to `0.03 m/s`
- out-of-plane relative velocity:
  - `0 m/s` for the 3-DoF scenario
  - `-0.03 m/s` to `0.03 m/s` for the 6-DoF scenario
- attitude error: `-5 deg` to `5 deg` on each axis
- angular velocity: `-0.10 deg/s` to `0.10 deg/s` on each axis

Motivation: The training range is wider than the testing range in order to expose the controller to a broader set of initial conditions during learning. The wider bounds help improve robustness while remaining physically consistent with a close-range post-homing rendezvous state. The spread is intentionally limited so that the sampled states remain representative of P2 and do not overlap with conditions that would correspond to earlier rendezvous phases.

### P3 definition

At P3, the chaser is assumed to have already completed the closing phase and to be ready for terminal docking. The state at P3 must therefore be significantly tighter than at P2, so that the terminal docking controller can focus on precise corridor tracking and terminal constraint satisfaction.

#### Nominal P3 state

- axial distance from the docking point: `300 m`
- lateral offset: `0 m`
- out-of-plane offset:
  - `0 m` for the 3-DoF scenario
  - `0 m` nominal for the 6-DoF scenario
- axial relative velocity: `-0.03 m/s`
- lateral relative velocity: `0 m/s`
- out-of-plane relative velocity:
  - `0 m/s` for the 3-DoF scenario
  - `0 m/s` nominal for the 6-DoF scenario
- attitude error: `0 deg` on each axis
- angular velocity: `0 deg/s` on each axis

Motivation: A nominal axial distance of 300 m is chosen to place the chaser close enough to the target to justify a dedicated terminal docking controller, while still leaving sufficient room for final approach and corridor tracking. Zero nominal lateral, out-of-plane, attitude, and angular-rate errors represent the ideal handover state from the closing controller to the terminal controller. The smaller nominal closing velocity reflects the fact that the approach speed should already have been reduced before the terminal docking phase begins.

#### P3 testing range

- axial distance: `280 m` to `320 m`
- lateral offset: `-3 m` to `3 m`
- out-of-plane offset:
  - `0 m` for the 3-DoF scenario
  - `-3 m` to `3 m` for the 6-DoF scenario
- axial relative velocity: `-0.05 m/s` to `-0.01 m/s`
- lateral relative velocity: `-0.005 m/s` to `0.005 m/s`
- out-of-plane relative velocity:
  - `0 m/s` for the 3-DoF scenario
  - `-0.005 m/s` to `0.005 m/s` for the 6-DoF scenario
- attitude error: `-1 deg` to `1 deg` on each axis
- angular velocity: `-0.02 deg/s` to `0.02 deg/s` on each axis

Motivation: The testing range at P3 is selected to represent a realistic handover state from which the terminal controller can begin precise docking. The lateral and out-of-plane offsets are made much smaller than at `P2`, reflecting that the closing controller should already have centered the chaser relative to the docking corridor. The tighter velocity, attitude, and angular-rate bounds reflect the more restrictive nature of the final approach phase.

#### P3 training range

- axial distance: `250 m` to `350 m`
- lateral offset: `-5 m` to `5 m`
- out-of-plane offset:
  - `0 m` for the 3-DoF scenario
  - `-5 m` to `5 m` for the 6-DoF scenario
- axial relative velocity: `-0.06 m/s` to `-0.01 m/s`
- lateral relative velocity: `-0.01 m/s` to `0.01 m/s`
- out-of-plane relative velocity:
  - `0 m/s` for the 3-DoF scenario
  - `-0.01 m/s` to `0.01 m/s` for the 6-DoF scenario
- attitude error: `-2 deg` to `2 deg` on each axis
- angular velocity: `-0.03 deg/s` to `0.03 deg/s` on each axis

Motivation: The training range at P3 is wider than the testing range in order to improve robustness of the terminal docking controller. Even though the range is wider, it remains significantly tighter than the corresponding range at P2, because P3 must still represent a near-aligned corridor-entry state. This preserves the physical distinction between the closing phase and the terminal docking phase.


## Environmental disturbances (Starin and Eterno, 2011)

In the initial version of the docking model, environmental disturbances are neglected. The emphasis is placed on the nominal close-proximity relative motion dynamics and on the correct implementation of the docking scenario, controller structure, reward design, phase handover logic, and terminal conditions. Under this assumption, the translational and rotational motion of the chaser is driven only by the nominal relative-motion model and the applied control inputs. This disturbance-free case is used as the baseline case for verification and performance evaluation.

### Disturbed environment

After the nominal docking model has been implemented and validated, the environment is extended to include the main external disturbance torques acting on the chaser. For typical Earth-orbiting spacecraft, the main environmental disturbance sources are gravity-gradient torque, magnetic torque, solar-radiation-pressure torque, and, in low-Earth orbit, aerodynamic torque. Since the present scenario is based on an ISS-like low-Earth orbit at about 400 km altitude, these four disturbance sources are taken as the main environmental effects to be included in the extended model: aerodynamic drag torque, magnetic torque, gravity-gradient torque, solar-radiation-pressure torque. 

The total environmental disturbance torque can then be modeled as

`tau_env = tau_aero + tau_mag + tau_gg + tau_srp`, where each contribution is computed separately.

#### 1. Atmospheric drag torque

At 400 km altitude, atmospheric drag is relevant because the residual atmosphere in LEO can generate a force in the ram direction. When the center of aerodynamic pressure is not aligned with the center of mass, this force produces a torque.

A suitable simplified implementation is:

`tau_aero = 0.5 * rho(h) * C_d * A_ram * V_rel^2 * l_aero`

where:

- `rho(h)` is the atmospheric density at altitude `h`
- `C_d` is the drag coefficient
- `A_ram` is the projected area normal to the velocity direction
- `V_rel` is the orbital-relative flow speed
- `l_aero` is the moment arm between the center of aerodynamic pressure and the center of mass

At 400 km, this term should generally be included before higher-altitude-only effects, because aerodynamic torque is one of the characteristic LEO disturbances. 

#### 2. Magnetic torque

Magnetic effects are relatively strong in LEO and the disturbance torque is modeled as the interaction between the spacecraft residual magnetic dipole moment and the Earth magnetic field. The Earth field can be approximated as a dipole for design purposes. 

A practical model is:

`tau_mag = m_res x B_E`

where:

- `m_res` is the spacecraft residual magnetic dipole vector
- `B_E` is the Earth magnetic field at the spacecraft location

For an initial implementation, `B_E` can be taken from a simple dipole model at the target orbit altitude and inclination. If a simpler first approximation is needed, the magnitude of `B_E` may be fixed and only its direction varied along the orbit.

#### 3. Gravity-gradient torque

Gravity-gradient torque arises because the gravity field is not perfectly uniform across the spacecraft body, and its simplified expression depends on the spacecraft inertia distribution, attitude relative to the local vertical, and orbit radius. 

A standard simplified model is:

`tau_gg = 3 * mu / r^3 * (e_r x J e_r)`

where:

- `mu` is the Earth gravitational parameter
- `r` is the orbital radius
- `e_r` is the local vertical unit vector expressed in the spacecraft frame
- `J` is the spacecraft inertia matrix

#### 4. Solar radiation pressure torque

SRP torque is modeled through the illuminated area, reflectance, incidence angle, and the offset between center of pressure and center of mass. It also notes that accurate SRP prediction is difficult, so a simplified estimate is appropriate at first. 

A suitable first model is:

`tau_srp = (Phi / c) * A_sun * (1 + q) * cos(phi) * l_srp`

where:

- `Phi` is the solar flux
- `c` is the speed of light
- `A_sun` is the illuminated projected area
- `q` is an effective reflectance factor
- `phi` is the incidence angle
- `l_srp` is the moment arm between center of pressure and center of mass

In this disturbed case, the translational and rotational dynamics are propagated using the nominal control inputs together with the environmental disturbance terms.

### Translational dynamics

The translational motion of the chaser center of mass is written as

`r_dot = v`

`m * v_dot = F_control + F_env`

where:

- `r` is the relative position of the chaser
- `v` is the relative velocity of the chaser
- `m` is the chaser mass
- `F_control` is the commanded control force
- `F_env` is the total environmental disturbance force

If the disturbance model is restricted to environmental torques only, then

`F_env = 0`

and the translational dynamics reduce to

`r_dot = v`

`m * v_dot = F_control`

In this case, the translational motion remains unchanged with respect to the disturbance-free model, and the environmental effects act only through the rotational dynamics.

### Rotational dynamics

The rotational motion of the chaser is written as

`J * omega_dot + omega x (J * omega) = tau_control + tau_env`

where:

- `J` is the chaser inertia matrix
- `omega` is the chaser angular velocity
- `tau_control` is the commanded control torque
- `tau_env` is the total environmental disturbance torque

The rotational dynamics are driven by their combined effect through the single total environmental torque `tau_env`.

## Performance conditions 

## Safety requirements 


## Assumptions 

Assumption: One spacecraft is modelled as the active chaser, responsible for performing the approach and control actions.
Consequence: 
Limitation:

Assumption: The other spacecraft is modelled as a passive target, with no active control input during the docking phase.
Consequence: 
Limitation:

Assumption: The target attitude is assumed to remain fixed throughout the modeled motion, and the target angular velocity is assumed to be zero.
Consequence: 
Limitation:

Assumption: Phasing part (earlier rendezvous maneuvers during which the large orbital differences between the chaser and target, such as relative distance, phase angle, and orbital plane differences, are corrected) and the homing part (chaser has already autonomously acquired the target orbit and reduced the relative velocity, reaching the station-keeping point P2, located a couple of kilometers away from the target) have already been completed successfully before the start of the modeled motion, therefore, the modeled motion begins from P2. (Luo, J. Zhang, and Tang, 2014)
Consequence: 
Limitation:

Assumption: The control task terminates at first contact between the chaser and the target (contact dynamics, latching mechanisms, and other capture-related processes are outside the scope of this control problem). 
Consequence: 
Limitation:

Assumption: The docking axis is assumed to coincide with the x_T-axis of the target-centered reference frame. 
Consequence: 
Limitation:

Assumption: The nominal docking corridor is assumed to be aligned with the x_T-axis of the target-centered reference frame. 
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

Assumption: The target orbit is assumed to be an ISS-like circular LEO with altitude 400 km, inclination 51.6 deg, eccentricity equal to zero, and right ascension of the ascending node, argument of perigee, and initial true anomaly fixed and set to zero. 
Consequence: 
Limitation:

Assumption: The close-proximity docking problem is assumed to be sufficiently local that the orbital angles do not significantly affect the modeled docking scenario.
Consequence: 
Limitation:

Assumption: The chaser is assumed to start on the positive x_T side of the target.
Consequence: 
Limitation:

Assumption: Nominal states are assumed to be representative reference conditions for both P2 and P3, and actual initial conditions are assumed to be generated by pertubing these nominal states within bounded intervals. 
Consequence: 
Limitation:

Assumption: A narrower testing range is assumed to represent realistic operational conditions, and a wider training range is assumed to improve controller robustness by exposing it to a broader set of initial conditions. 
Consequence: 
Limitation:

Assumption: Environmental disturbances are neglected in the initial docking model.
Consequence: 
Limitation:

Assumption: After validation of the nominal model, the docking environment is assumed to be extendable to a disturbed case represented by the main external environmental disturbance torques: aerodynamic drag torque, magnetic torque, gravity-gradient torque, solar-radiation-pressure torque.
Consequence: 
Limitation:

Assumption: The aerodynamic disturbance is assumed to act through a ram-direction force that produces a torque when the center of aerodynamic pressure is not aligned with the center of mass.
Consequence: 
Limitation:

Assumption: The Earth magnetic field is assumed to be approximable by a dipole model.
Consequence: 
Limitation:

Assumption: The spacecraft is assumed to possess a non-zero residual magnetic dipole moment.
Consequence: 
Limitation:

Assumption: The gravity-gradient torque is assumed to arise from the non-uniformity of the gravity field across the spacecraft body.
Consequence: 
Limitation:

Assumption: The SRP torque is assumed to arise from an offset between the center of pressure and the center of mass.
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

Assumption: 
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





