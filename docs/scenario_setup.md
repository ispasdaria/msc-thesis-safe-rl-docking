# General 6-DoF Scenario Setup

## 1. Purpose and Scope

This document defines the parent rendezvous-and-docking scenario as a three-dimensional, six-degree-of-freedom (6-DoF) problem. It specifies the mission phases, reference frames, vehicle and port geometry, initialization ranges, dynamics, safety sets, terminal conditions, and general reward objectives.

The implemented planar reduction is intentionally documented separately in [`src/envs/docking3dof/simplified3dofscenario.md`](../src/envs/docking3dof/simplified3dofscenario.md). That document identifies every state, dynamics, geometry, actuation, safety, reward, and episode-logic modification made for the current 3-DoF environment. Implementation-specific values should not be inferred from this document unless the simplified-scenario document explicitly adopts them.

The modeled mission contains two sequential close-proximity tasks:

1. Closing phase: P2 -> P3
2. Final-approach phase: P3 -> docking

![Docking phases](docking_phases.png)

The split is motivated by phase-specific objectives and constraints:

- Closing emphasizes convergence from larger range, relative-velocity reduction, and preparation for corridor entry.
- Final approach emphasizes corridor adherence, interface alignment, low contact speed, and terminal precision.

### 1.1 Controller 1: Corridor Capture

Controller 1 transfers the chaser from P2, approximately 1.5--1.9 km from the target CoM in the adopted initialization set, to P3, approximately 250--350 m from the target CoM.

Its objectives are to:

- reduce range to the selected P3 handover state;
- reduce relative velocity;
- reduce lateral and out-of-plane displacement;
- prepare position, velocity, attitude, and angular rate for final approach.

### 1.2 Controller 2: Terminal Docking

Controller 2 starts at P3 and guides the chaser through the docking corridor to an admissible contact state.

Its objectives are to:

- enter and track the nominal approach corridor;
- keep lateral and out-of-plane port error small;
- reduce approach speed near contact;
- align the two docking interfaces;
- satisfy the terminal position, velocity, attitude, and angular-rate constraints.

## 2. Vehicles and Docking Geometry

### 2.1 Core Geometry

- `e_d` is the target docking-axis unit vector. It is normal to the target docking-interface plane and points outward from the target.
- The target docking-port center is at a fixed offset from the target CoM.
- The chaser docking-port center is at a fixed offset from the chaser CoM in the chaser body frame.
- The nominal docking attitude is face-to-face: the target port normal points toward the approaching chaser and the chaser port normal points toward the target.

### 2.2 Docking-Port Relative Kinematics

Let:

- `r in R^3` be the chaser-CoM position relative to the target CoM, expressed in the target-centered orbital frame;
- `v in R^3` be the corresponding relative CoM velocity;
- `q_e` be the chaser attitude error relative to the nominal face-to-face attitude;
- `R_Bc_to_O(q_e)` be the actual rotation from the chaser body frame to the target-centered orbital frame, including the nominal face-to-face orientation;
- `r_c in R^3` be the chaser-port offset from the chaser CoM, expressed in the chaser body frame;
- `r_t in R^3` be the target-port offset from the target CoM, expressed in the target-centered orbital frame.

The docking-port relative position is

```
r_p = r + R_Bc_to_O(q_e) r_c - r_t
```

and exact geometric coincidence corresponds to

```
r_p = 0.
```

If the target is fixed relative to the selected target-centered frame, the port-relative velocity is

```
v_p = v + omega_c x (R_Bc_to_O(q_e) r_c),
```

where `omega_c` is the chaser angular velocity relative to that frame and `x` denotes the vector cross product. If target attitude motion is introduced, the target-port rotational velocity and the required frame-transport terms must also be subtracted.

Terminal conditions are imposed at docking-port level because CoM coincidence alone does not describe interface contact quality.

### 2.3 Vehicle Reference Cases

The geometry uses reference cases from the International Docking System Standard (IDSS), Revision E.

Baseline case:

- Chaser: Starliner-like visiting vehicle, approximated by IDSS-15T.
- Target: ISS-like orbital complex, approximated by IDSS-350T.

Alternative vehicle-to-vehicle case:

- Chaser: Starliner-like visiting vehicle, approximated by IDSS-15T.
- Target: Crew Dragon-like vehicle, approximated by IDSS-10T.

These are project modeling choices, not official IDSS classifications of the named vehicles. IDSS Revision E Table 3.3.1.2-1 gives the following Hard Capture System mating-plane-center coordinates relative to vehicle CG, with positive x toward the docking interface:

- IDSS-15T: `[4.1, 0, 0] m`
- IDSS-10T: `[3.5, 0, 0] m`
- IDSS-350T: `[20, 0, 2] m`

The full 6-DoF geometry retains all three components of the selected reference offset. Vehicle mass, inertia, thruster layout, and other parameters must be supplied independently when a force-and-torque model is instantiated.

## 3. Reference Frames

### 3.1 Earth-Centered Inertial Frame `F_I`

`F_I` is used for absolute orbital motion.

- Origin: Earth CoM
- `Z_I`: Earth rotation axis toward the North Pole
- `X_I`: toward the vernal equinox
- `Y_I`: completes the right-handed equatorial triad

### 3.2 Target Body-Fixed Frame `F_{B_t}`

`F_{B_t}` is rigidly attached to the target.

- Origin: target CoM
- `x_{B_t}`: toward the target docking interface; nominally aligned with `+x_O`
- `z_{B_t}`: downward in the target symmetry plane
- `y_{B_t}`: completes the right-handed triad

### 3.3 Target-Centered Orbital Frame `F_O`

`F_O` is used for relative translational motion.

- Origin: target CoM
- `x_O`: along-track and along the outward docking direction, approximately the target velocity direction (V-bar)
- `z_O`: local radial direction from the target toward Earth
- `y_O = z_O x x_O`: completes the right-handed triad

With this explicit sign convention, `y_O` is opposite the conventional positive orbital-angular-momentum direction for a prograde orbit. Stating the construction avoids ambiguity between different LVLH and H-bar conventions.

![Orbital reference frame](orbital_reference_frame.png)

### 3.4 Chaser Body-Fixed Frame `F_{B_c}`

`F_{B_c}` is rigidly attached to the chaser.

- Origin: chaser CoM
- `x_{B_c}`: toward the chaser docking interface; at nominal face-to-face alignment it points along `-x_O`
- `z_{B_c}`: downward in the chaser symmetry plane
- `y_{B_c}`: completes the right-handed triad

![Target and chaser body-centered reference frames](body-centered_reference_frames.png)

The frame hierarchy separates absolute orbital motion (`F_I`), target-relative translation (`F_O`), and vehicle-fixed geometry and actuation (`F_{B_t}` and `F_{B_c}`).

## 4. Orbit and Reference Handover Geometry

### 4.1 Target Orbit

The baseline target follows an ISS-like circular low-Earth orbit:

- altitude: `400 km`;
- inclination: `51.6 deg`;
- eccentricity: `0`;
- semi-major axis: `a = R_E + 400 km`;
- RAAN: `0 deg` at the reference epoch;
- argument of perigee: `0 deg`;
- true anomaly: `0 deg` at the reference epoch.

The circular baseline supports a clean local orbital frame and avoids ellipticity effects during initial controller development.

### 4.2 Reference Docking Geometry

- The target docking axis is aligned with `+x_O`.
- The chaser starts on the positive `x_O` side and approaches toward decreasing `x_O`.
- The target attitude is fixed relative to `F_O` in the baseline scenario.
- The target angular velocity relative to `F_O` is zero in the baseline scenario.

### 4.3 Handover Points

- P2 is the controlled close-range state after phasing and homing.
- P3 is the tighter corridor-entry state and the handover to terminal approach.

P2 and P3 numerical positions below are chaser-CoM coordinates. Port-relative quantities are calculated using the vehicle offsets and attitude.

## 5. Nominal States and Randomized Initialization

The nominal state defines a reference case. Testing ranges represent narrower operational evaluation, while training ranges provide wider robustness exposure.

For the small initialization angles used here, attitude-error bounds are specified component-wise about the three body axes. A quaternion or rotation matrix should be used internally to avoid singularities during propagation.

### 5.1 P2 Initialization

#### Nominal P2 State

- CoM axial coordinate: `r_x = 1700 m`
- Lateral coordinate: `r_y = 0 m`
- Out-of-plane coordinate: `r_z = 0 m`
- Axial relative velocity: `v_x = -0.10 m/s`
- Lateral relative velocity: `v_y = 0 m/s`
- Out-of-plane relative velocity: `v_z = 0 m/s`
- Attitude error: `0 deg` about each axis
- Angular velocity: `0 deg/s` about each axis

#### P2 Testing Range

- `r_x in [1600, 1800] m`
- `r_y in [-100, 100] m`
- `r_z in [-100, 100] m`
- `v_x in [-0.15, -0.05] m/s`
- `v_y in [-0.02, 0.02] m/s`
- `v_z in [-0.02, 0.02] m/s`
- attitude error: `[-3, 3] deg` about each axis
- angular velocity: `[-0.05, 0.05] deg/s` about each axis

#### P2 Training Range

- `r_x in [1500, 1900] m`
- `r_y in [-150, 150] m`
- `r_z in [-150, 150] m`
- `v_x in [-0.20, -0.03] m/s`
- `v_y in [-0.03, 0.03] m/s`
- `v_z in [-0.03, 0.03] m/s`
- attitude error: `[-5, 5] deg` about each axis
- angular velocity: `[-0.10, 0.10] deg/s` about each axis

### 5.2 P3 Initialization

#### Nominal P3 State

- CoM axial coordinate: `r_x = 300 m`
- Lateral coordinate: `r_y = 0 m`
- Out-of-plane coordinate: `r_z = 0 m`
- Axial relative velocity: `v_x = -0.03 m/s`
- Lateral relative velocity: `v_y = 0 m/s`
- Out-of-plane relative velocity: `v_z = 0 m/s`
- Attitude error: `0 deg` about each axis
- Angular velocity: `0 deg/s` about each axis

#### P3 Testing Range

- `r_x in [280, 320] m`
- `r_y in [-3, 3] m`
- `r_z in [-3, 3] m`
- `v_x in [-0.05, -0.01] m/s`
- `v_y in [-0.01, 0.01] m/s`
- `v_z in [-0.005, 0.005] m/s`
- attitude error: `[-1, 1] deg` about each axis
- angular velocity: `[-0.05, 0.05] deg/s` about each axis

#### P3 Training Range

- `r_x in [250, 350] m`
- `r_y in [-5, 5] m`
- `r_z in [-5, 5] m`
- `v_x in [-0.06, -0.01] m/s`
- `v_y in [-0.02, 0.02] m/s`
- `v_z in [-0.01, 0.01] m/s`
- attitude error: `[-2, 2] deg` about each axis
- angular velocity: `[-0.10, 0.10] deg/s` about each axis

P3 ranges are intentionally tighter than P2 ranges so that the second controller begins from a corridor-entry neighborhood.

## 6. General 6-DoF Dynamics and Disturbances

### 6.1 State and Dynamics

The general state is

```
x = [r, v, q_e, omega_e],
```

where `r, v in R^3`, `q_e` is a unit attitude-error quaternion, and `omega_e in R^3` is the relative angular velocity.

The inertial translational equations are written schematically as

```
r_I_dot = v_I
m v_I_dot = F_gravity + F_control + F_environment.
```

Transforming these equations into `F_O` must retain the appropriate relative-gravity, Coriolis, centrifugal, and frame-angular-acceleration terms. The chosen high-fidelity propagator or relative-motion approximation must be stated when the 6-DoF model is implemented.

The chaser rotational dynamics are

```
J omega_dot + omega x (J omega) = tau_control + tau_environment.
```

Quaternion kinematics propagate the attitude, with quaternion normalization maintained numerically.

### 6.2 Control Input

The physical 6-DoF control input is

```
u_Bc = [F_Bc, tau_Bc],
```

where `F_Bc in R^3` and `tau_Bc in R^3` are the commanded chaser-body force and torque. The admissible set must reflect the selected vehicle's thruster geometry, force limits, torque limits, minimum impulse, duty cycle, and allocation logic. If a guidance-level specific-force interface is used instead, that modeling choice and its relationship to mass and inertia must be documented in the simplified or implemented model.

### 6.3 Environmental Disturbances

The nominal baseline neglects environmental disturbances so that scenario, handover, controller, and safety logic can first be verified under controlled conditions.

A disturbed extension may include

```
tau_environment = tau_aero + tau_mag + tau_gg + tau_srp
```

and the corresponding environmental forces where applicable.

Aerodynamic disturbance:

```
F_aero = -0.5 rho(h) C_d A_ram ||V_rel||^2 e_Vrel
tau_aero = r_CoP,aero x F_aero
```

Magnetic disturbance:

```
tau_mag = m_res x B_E
```

Gravity-gradient disturbance:

```
tau_gg = 3 mu / r_orbit^3 (e_r x J e_r)
```

Solar-radiation-pressure disturbance:

```
F_srp = (Phi/c) A_sun (1 + q_srp) cos(phi) e_sun
tau_srp = r_CoP,srp x F_srp
```

The density, magnetic-field, eclipse, optical-property, center-of-pressure, mass, and inertia models must be specified before this extension is treated as quantitative.

## 7. General Docking Constraints

### 7.1 Derived Quantities

With `e_d = [1, 0, 0]` in `F_O`, define

```
s = e_d^T r_p
rho = ||(I - e_d e_d^T) r_p||
s_dot = e_d^T v_p.
```

Here `s` is port-axial separation and `rho` is the two-dimensional transverse port offset.

### 7.2 Primitive Geometric Sets

Approach ellipsoid with full axis lengths `4 x 2 x 2 km`:

```
E_AE = {r : r^T Q_AE r <= 1}
Q_AE = diag(1/2000^2, 1/1000^2, 1/1000^2).
```

Keep-out sphere:

```
S_KOS = {r : ||r|| <= 200 m}.
```

Docking cone:

```
C_cone = {x : 0 <= s <= 2000 m and rho <= 0.1 s}.
```

Docking corridor:

```
C_corr = {x : 0 <= s <= d_P3 and rho <= 5 m}.
```

The ellipsoid and keep-out sphere are CoM-based; the cone and corridor are port-based.

`d_P2` and `d_P3` denote the port-axial separations associated with the selected P2 and P3 reference states. An implementation must state whether they are fixed nominal values or recomputed from sampled states.

![Scheme of the docking geometric constraints](docking_geometric_constraints.jpeg)


### 7.3 Alignment Sets

Let `theta_e` be the principal attitude-error angle obtained from the relative quaternion, and let `||omega_e||` be the relative angular-rate magnitude.

```
A_safe = {x : theta_e <= 3 deg and ||omega_e|| <= 0.15 deg/s}

A_warn = {x : theta_e <= 4 deg and ||omega_e|| <= 0.20 deg/s} \ A_safe

A_unsafe = {x : theta_e > 4 deg or ||omega_e|| > 0.20 deg/s}.
```

### 7.4 Velocity Sets

Closing:

```
V_safe_close = {x : ||v_p|| <= 0.25 m/s and |s_dot| <= 0.15 m/s}

V_warn_close =
    {x : ||v_p|| <= 0.35 m/s and |s_dot| <= 0.25 m/s}
    \ V_safe_close

V_unsafe_close = {x : ||v_p|| > 0.35 m/s or |s_dot| > 0.25 m/s}.
```

Final approach:

```
V_safe_final = {x : ||v_p|| <= 0.15 m/s and |s_dot| <= 0.10 m/s}

V_warn_final =
    {x : ||v_p|| <= 0.20 m/s and |s_dot| <= 0.15 m/s}
    \ V_safe_final

V_unsafe_final = {x : ||v_p|| > 0.20 m/s or |s_dot| > 0.15 m/s}.
```

### 7.5 Terminal Docking Set

The general terminal acceptance set is

```
X_dock = {
    x : ||r_p|| <= 0.10 m,
        ||v_p|| <= 0.10 m/s,
        theta_e <= 4 deg,
        ||omega_e|| <= 0.20 deg/s
}.
```

This is a guidance-level contact tolerance. It does not represent soft capture, latching, hard capture, contact loads, or the complete IDSS capture envelope.

### 7.6 Phase Safety Sets

Closing geometry:

```
G_safe_close = (E_AE \ S_KOS) intersect C_cone
               intersect {d_P3 <= s <= d_P2}

G_unsafe_close = (S_KOS \ C_cone) union complement(E_AE)

G_warn_close =
      (E_AE \ (S_KOS union C_cone))
    union (S_KOS intersect C_cone)
    union (E_AE intersect C_cone intersect {s < d_P3}).
```

Closing combined sets:

```
X_safe_close = G_safe_close intersect V_safe_close
X_unsafe_close = G_unsafe_close union V_unsafe_close
X_warn_close = (G_warn_close union V_warn_close) \ X_unsafe_close.
```

Final-approach geometry:

```
G_safe_final = C_cone intersect C_corr
G_unsafe_final = complement(C_cone)
G_warn_final = C_cone \ C_corr.
```

Final-approach combined sets:

```
X_safe_final = G_safe_final intersect V_safe_final intersect A_safe
X_unsafe_final = G_unsafe_final union V_unsafe_final union A_unsafe
X_warn_final =
    (G_warn_final union V_warn_final union A_warn) \ X_unsafe_final.
```

Warning states are admissible but undesirable. Unsafe states terminate the maneuver. A complete implementation must ensure that every reachable state receives an unambiguous safe, warning, or unsafe classification.

### 7.7 Phase Definitions

Closing:

- initial state: `x(t_P2) in X_P2`;
- admissible evolution: `x(t) not in X_unsafe_close` and `u(t) in U`;
- successful handover: `x(t_f)` satisfies the selected P3 target/tolerance definition.

Final approach:

- initial state: `x(t_P3) in X_P3`;
- admissible evolution: `x(t) not in X_unsafe_final` and `u(t) in U`;
- successful completion: `x(t_f) in X_dock` without a hard safety violation.

## 8. General Reward and Episode Objectives

The reward is phase-dependent. Exact coefficients belong to each implemented controller and are therefore documented with that implementation.

### 8.1 Closing Objective

A closing reward should balance

```
r_close =
    - w_P3 d_P3_error
    - w_v ||v_p||
    - w_rho rho
    - w_att theta_e
    - w_rate ||omega_e||
    - w_u control_effort
    + w_progress progress_to_P3
    - w_warn I_warn
    - w_fail I_fail
    + w_success I_P3.
```

### 8.2 Final-Approach Objective

A final-approach reward should balance

```
r_final =
    - w_rp ||r_p||
    - w_vp ||v_p||
    - w_rho rho
    - w_att theta_e
    - w_rate ||omega_e||
    - w_u control_effort
    + w_progress progress_to_contact
    - w_warn I_warn
    - w_fail I_fail
    + w_dock I_dock.
```

Progress incentives must not make unsafe motion more attractive than constraint adherence. Reward success is subordinate to the safety monitor: a state that violates a hard constraint is a failure even if it also satisfies a geometric target tolerance.

### 8.3 Episode Outcomes

- Success: the phase terminal set is reached without a hard safety violation.
- Hard failure: an unsafe constraint or non-finite numerical state is reached.
- Timeout: the phase horizon is reached without success; it is unsuccessful but should be distinguished from a physical safety failure.

## 9. General Assumptions and Limitations

1. The chaser is active and the target is passive during the modeled maneuver.
2. Phasing and homing are complete before P2.
3. The baseline target attitude is fixed relative to `F_O`.
4. The nominal docking axis is aligned with `+x_O`.
5. The chaser approaches from positive `x_O` toward decreasing `x_O`.
6. P2 and P3 initialization positions are CoM-based; docking acceptance is port-based.
7. The IDSS vehicle cases provide reference geometry and do not by themselves define all dynamic or actuator parameters.
8. The baseline uses a circular ISS-like orbit.
9. The nominal environment neglects disturbances; the disturbed model requires additional quantitative parameters.
10. Contact dynamics, soft capture, latching, hard capture, and structural loads are outside the guidance-level terminal model.
11. Training ranges are wider than testing ranges to expose the controller to off-nominal states.
12. A future 6-DoF implementation must document its orbital propagator, attitude convention, vehicle mass and inertia, actuator model, numerical integrator, timestep, observation definition, action scaling, phase horizons, and exact reward coefficients.

## 10. Sources and Version Notes

- [International Docking System Standard, Interface Definition Document, Revision E, October 2016](https://www.internationaldockingstandard.com/download/IDSS_IDD_Revision_E.pdf), especially Section 3.3.1.2 and Table 3.3.1.2-1.
- Oestreich, Linares, and Gondhalekar (2020) motivates port-level relative kinematics and reinforcement-learning-based docking control.
- Starin and Eterno (2011) provides standard spacecraft attitude-control disturbance categories.
- The docking-phase figure is attributed to Luo, Zhang, and Tang (2014).

The numerical and logical deviations used by the current planar environment are maintained in the simplified-scenario document rather than in this general specification.
