# Implemented Simplified 3-DoF Scenario

## 1. Purpose and Relationship to the General Scenario

This document defines the planar environment implemented by `core.py` and `gym_env.py` in this directory. It is the implementation-specific companion to the [general 6-DoF scenario](../../../docs/scenario_setup.md).

The general scenario remains the parent mission definition. This file records what is retained, projected, replaced, or omitted in order to obtain the current 3-DoF model. When an implementation detail differs from the general formulation, this document and the executable code describe the current environment.

| General 6-DoF element | Implemented 3-DoF treatment |
|---|---|
| Three relative-position and velocity components | Retains `x_O-y_O`; omits the third translational axis |
| Three-axis attitude and angular velocity | Retains scalar attitude error `psi` and rate `omega_z` about `z_O` |
| Unit-quaternion attitude propagation | Replaced by a wrapped scalar angle |
| Full IDSS port offsets | Uses planar projections `[4.1, 0] m` and `[20, 0] m` |
| Force and torque actuation | Replaced by directly commanded linear and angular accelerations |
| Vehicle mass, inertia, and thruster allocation | Not modeled |
| Absolute ECI/orbital propagation | Replaced by reduced target-relative planar equations |
| 3D ellipsoid, sphere, cone, and corridor | Uses their planar ellipse, circle, wedge, and strip counterparts |
| 3D attitude and rate norms | Uses `|psi|` and `|omega_z|` |
| Environmental forces and torques | Omitted |
| Physical contact and capture | Replaced by a port-level numerical acceptance tolerance |

The mission remains split into closing (P2 -> P3) and final approach (P3 -> docking), with one policy trained for each phase.

## 2. Planar State, Frames, and Port Kinematics

### 2.1 State

The simulator state is

```
x_3DoF = [rx, ry, vx, vy, psi, omega_z].
```

- `[rx, ry]` is the chaser-CoM position relative to the target CoM in the `x_O-y_O` plane.
- `[vx, vy]` is the corresponding relative CoM velocity.
- `psi` is the chaser attitude error about `z_O`, wrapped to `[-pi, pi)` where required.
- `omega_z` is the corresponding relative angular velocity.

The omitted states are `rz`, `vz`, two attitude-error components, and two angular-rate components. They are not constrained to evolve near zero; they do not exist in the state model. Consequently, the environment cannot represent out-of-plane escape, roll/pitch misalignment, or coupling through the omitted axes.

### 2.2 Nominal Face-to-Face Convention

The target docking axis is `+x_O`. The chaser begins on the positive `x_O` side and approaches toward decreasing `x_O`. At `psi = 0`, the chaser body `+x` axis points along `-x_O`, toward the target.

The planar body-to-orbital rotation is

```
R_Bc_to_O(psi) =
    [[-cos(psi),  sin(psi)],
     [-sin(psi), -cos(psi)]].
```

The minus signs encode the nominal face-to-face orientation; `psi = 0` is aligned, not anti-aligned.

### 2.3 Port Offsets

The selected general vehicle case is projected into the plane:

```
r_c_body    = [4.1, 0] m
r_t_orbital = [20.0, 0] m.
```

The chaser uses the IDSS-15T reference offset. The target uses the in-plane part of the IDSS-350T offset `[20, 0, 2] m`; its `2 m` third-axis component is omitted.

### 2.4 Port-Relative Quantities

The port-relative position is

```
r_p = [rx, ry] + R_Bc_to_O(psi) r_c_body - r_t_orbital.
```

With

```
S = [[0, -1],
     [1,  0]],
```

the port-relative velocity is

```
v_p = [vx, vy] + omega_z S R_Bc_to_O(psi) r_c_body.
```

The target port is stationary in this relative-frame model. Target rotation and additional frame-transport terms are not included.

The derived docking quantities are

```
s     = r_p,x
rho   = |r_p,y|
s_dot = v_p,x.
```

Thus, `s` and `rho` are port-based, while the stored position state and the approach ellipse/keep-out circle are CoM-based.

## 3. Implemented Dynamics and Numerical Propagation

### 3.1 Orbit Reduction

The general ISS-like orbit is reduced to two constants:

```
mu = 3.986004418e14 m^3/s^2
r0 = 6,778,000 m
n  = sqrt(mu / r0^3).
```

Inclination, RAAN, argument of perigee, true anomaly, absolute target position, and absolute chaser position are not propagated.

### 3.2 Equations of Motion

Let `[a_x_O, a_y_O, alpha_z]` be the body command after rotation into the orbital frame. The implemented equations are

```
r_chaser = sqrt(r0^2 + rx^2 + ry^2)

rx_dot = vx
ry_dot = vy

vx_dot = -mu rx / r_chaser^3 + n^2 rx + a_x_O
vy_dot = -mu ry / r_chaser^3          + a_y_O

psi_dot     = omega_z
omega_z_dot = alpha_z.
```

These are reduced nonlinear relative equations. They do not implement a complete 3D LVLH model: the omitted radial state and the full Coriolis, centrifugal, and frame-coupling structure of a general rotating-frame formulation are absent.

No mass or inertia appears because the action is already an effective acceleration command. Environmental forces and torques are zero.

### 3.3 Integration and Time

- Default timestep: `1 s`
- Default integrator: fourth-order Runge--Kutta (`RK4`)
- Optional integrator: forward Euler
- Default Gymnasium horizon for closing: `12,000 s`
- Default Gymnasium horizon for final approach: `5,000 s`
- Direct construction of the lower-level core uses `10,800 s` unless overridden

The horizon can be changed through `max_time`.

## 4. Implemented Action

### 4.1 Physical Action

The core action is a chaser-body-frame acceleration command:

```
u_Bc = [a_x_Bc, a_y_Bc, alpha_z_Bc].
```

It is clipped component-wise to

```
-0.01  <= a_x_Bc   <= 0.01  m/s^2
-0.01  <= a_y_Bc   <= 0.01  m/s^2
-0.003 <= alpha_z  <= 0.003 rad/s^2.
```

The translational admissible set is therefore a box in the rotating chaser body frame, not a Euclidean-norm ball. A simultaneous maximum command on both axes has magnitude `sqrt(2) * 0.01 = 0.0141 m/s^2`.

The translational command is rotated with `R_Bc_to_O(psi)` before use in the dynamics. At `psi = 0`, positive chaser-body x acceleration maps to negative `x_O` acceleration.

### 4.2 Gymnasium Action

With the default normalized interface, the agent action lies in `[-1, 1]^3` and is scaled component-wise by

```
[0.01, 0.01, 0.003].
```

The environment does not model thruster selection, minimum impulse, pulse-width modulation, propellant, delays, mass variation, or actuator allocation.

## 5. Implemented Initialization and Handover

All sampled positions below are CoM coordinates. Port position is derived afterward from the sampled attitude and port offsets.

### 5.1 P2 Reset Distribution

Nominal:

```
[rx, ry, vx, vy, psi, omega_z]
= [1700, 0, -0.10, 0, 0, 0].
```

Testing:

```
rx      in [1600, 1800] m
ry      in [-100, 100] m
vx      in [-0.15, -0.05] m/s
vy      in [-0.02, 0.02] m/s
psi     in [-3, 3] deg
omega_z in [-0.05, 0.05] deg/s.
```

Training:

```
rx      in [1500, 1900] m
ry      in [-150, 150] m
vx      in [-0.20, -0.03] m/s
vy      in [-0.03, 0.03] m/s
psi     in [-5, 5] deg
omega_z in [-0.10, 0.10] deg/s.
```

At nominal alignment, `rx = 1700 m` gives a port-axial separation of `1675.9 m` because

```
s = 1700 - 4.1 - 20 = 1675.9 m.
```

### 5.2 P3 Reset Distribution

Nominal:

```
[rx, ry, vx, vy, psi, omega_z]
= [300, 0, -0.03, 0, 0, 0].
```

Testing:

```
rx      in [280, 320] m
ry      in [-3, 3] m
vx      in [-0.05, -0.01] m/s
vy      in [-0.01, 0.01] m/s
psi     in [-1, 1] deg
omega_z in [-0.05, 0.05] deg/s.
```

Training:

```
rx      in [250, 350] m
ry      in [-5, 5] m
vx      in [-0.06, -0.01] m/s
vy      in [-0.02, 0.02] m/s
psi     in [-2, 2] deg
omega_z in [-0.10, 0.10] deg/s.
```

At nominal alignment, `rx = 300 m` gives `s = 275.9 m`.

Some extreme P3 training samples can begin outside the `5 m` port corridor because lateral safety is evaluated at port level after attitude and port-offset effects are applied. Such states are warnings when they remain inside the cone; they are not automatically hard failures.

### 5.3 Sampled Closing Target

A closing reset samples both an initial P2 state and a separate P3 target from the same selected mode. Closing does not succeed merely by entering the full P3 reset range. It succeeds when every state component is within the following tolerance of the episode's sampled target:

```
|rx - rx_target|             <= 20 m
|ry - ry_target|             <= 2 m
|vx - vx_target|             <= 0.01 m/s
|vy - vy_target|             <= 0.01 m/s
|wrap(psi - psi_target)|     <= 0.5 deg
|omega_z - omega_z,target|   <= 0.05 deg/s.
```

This set is denoted `X_P3,target`.

The episode-specific axial values are also computed from port geometry:

- closing `d_P2`: port-axial separation of the sampled initial state;
- closing `d_P3`: port-axial separation of the sampled P3 target;
- final `d_P3`: port-axial separation of the sampled initial P3 state.

## 6. Planar Safety Sets

### 6.1 Primitive Geometry

The general 3D sets are reduced as follows.

Approach ellipse, evaluated using CoM position:

```
E_AE = {(rx, ry) : (rx/2000)^2 + (ry/1000)^2 <= 1}.
```

Keep-out circle, evaluated using CoM position:

```
S_KOS = {(rx, ry) : sqrt(rx^2 + ry^2) <= 200 m}.
```

Docking wedge, called `C_dock` in the code and corresponding to the general docking cone:

```
C_cone = {x : 0 <= s <= 2000 m and rho <= 0.1 s}.
```

Final corridor strip:

```
C_corr = {x : 0 <= s <= d_P3 and rho <= 5 m}.
```

### 6.2 Planar Alignment Classification

```
A_safe = {x : |psi| <= 3 deg and |omega_z| <= 0.15 deg/s}

A_warn =
    {x : |psi| <= 4 deg and |omega_z| <= 0.20 deg/s}
    \ A_safe

A_unsafe = {x : |psi| > 4 deg or |omega_z| > 0.20 deg/s}.
```

Alignment is calculated in both phases. It affects hard safety only in final approach. During closing it is a diagnostic and reward/handover quantity, so even an alignment state that meets the final phase's `A_unsafe` criterion does not by itself terminate a closing episode.

### 6.3 Velocity Classification

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

### 6.4 Closing Geometry Classification

The code implements

```
G_safe_close =
    (E_AE \ S_KOS) intersect C_cone intersect {d_P3 <= s <= d_P2}

G_unsafe_close =
    (S_KOS \ C_cone) union complement(E_AE)

G_warn_close =
      (E_AE \ (S_KOS union C_cone))
    union (S_KOS intersect C_cone)
    union (E_AE intersect C_cone intersect {s < d_P3}).
```

The combined classifications are

```
X_safe_close   = G_safe_close intersect V_safe_close
X_unsafe_close = G_unsafe_close union V_unsafe_close
X_warn_close   = (G_warn_close union V_warn_close) \ X_unsafe_close.
```

Implementation limitation: a state in `(E_AE \ S_KOS) intersect C_cone` with `s > d_P2` is neither safe, warning, nor unsafe. It does not terminate the current environment. This unclassified edge should be resolved before the monitor is treated as an exhaustive safety specification.

### 6.5 Final-Approach Classification

```
G_safe_final   = C_cone intersect C_corr
G_unsafe_final = complement(C_cone)
G_warn_final   = C_cone \ C_corr

X_safe_final =
    G_safe_final intersect V_safe_final intersect A_safe

X_unsafe_final =
    G_unsafe_final union V_unsafe_final union A_unsafe

X_warn_final =
    (G_warn_final union V_warn_final union A_warn) \ X_unsafe_final.
```

Warnings are nonterminal in both phases and receive a reward penalty. Unsafe classifications terminate the episode as failures.

## 7. Terminal and Episode Logic

### 7.1 Numerical Contact Tolerance

`is_in_X_dock` checks

```
X_contact = {
    x : ||r_p|| <= 0.10 m,
        ||v_p|| <= 0.10 m/s,
        |psi| <= 4 deg,
        |omega_z| <= 0.20 deg/s
}.
```

The safety monitor is evaluated before success. The effective terminal-success set is therefore

```
X_dock,effective = X_contact intersect complement(X_unsafe_final).
```

This matters near the cone tip: a state may satisfy the Euclidean `0.10 m` contact tolerance yet be outside the wedge and consequently fail rather than succeed.

No collision response, compliant contact, capture, latching, loads, or rebound is simulated.

### 7.2 Outcome Order

After each propagated step, the core evaluates outcomes in this order:

1. hard safety failure;
2. phase success;
3. time limit;
4. continuation.

Closing success is membership in `X_P3,target`. Final success is membership in the effective docking set above.

A hard failure produces `terminated = true` and `failure = true`. A time limit produces `truncated = true`, applies the timeout penalty, and leaves `failure = false`.

## 8. Reinforcement-Learning Interface

### 8.1 Observations

The final-approach observation contains six port-level values:

```
[r_p,x, r_p,y, v_p,x, v_p,y, psi_e, omega_z].
```

The closing observation concatenates that six-element port state with the six-element error relative to the sampled P3 target:

```
[port_state,
 rx-rx_target,
 ry-ry_target,
 vx-vx_target,
 vy-vy_target,
 wrap(psi-psi_target),
 omega_z-omega_z,target].
```

The policy therefore observes port quantities used by safety and, during closing, CoM-state error used by the handover objective.

### 8.2 Closing Reward (Current v15)

Let `e = x - x_P3,target`, with its angular component wrapped. Define

```
d_pos   = ||e_position||
d_vel   = ||e_velocity||
e_y     = |e_lateral_position|
e_vy    = |e_lateral_velocity|
e_psi   = |e_attitude|
e_omega = |e_angular_rate|

delta_d     = previous(d_pos) - current(d_pos)
delta_e_psi = previous(e_psi) - current(e_psi)

D_s = clip((|s_dot| - 0.15) / (0.25 - 0.15), 0, 1)
D_v = clip((||v_p|| - 0.25) / (0.35 - 0.25), 0, 1)

I_no_progress = 1
    if d_pos > 50 m and delta_d < 0.01 m,
    else 0.
```

With `u_norm` denoting the component-wise normalized physical action,

```
r_close,base =
    - 0.002 d_pos
    - 0.5   d_vel
    - 0.01  e_y
    - 5.0   e_vy
    - 5.0   e_psi
    + 50.0  delta_e_psi
    - 0.1   e_omega
    - 0.01  ||u_norm||
    + 20.0  delta_d
    - 20.0  D_s^2
    - 20.0  D_v^2
    - 0.05  I_no_progress.
```

The `20 delta_d` term is symmetric: progress is rewarded and regression is penalized with the same coefficient. Axial overshoot remains a diagnostic but contributes zero reward in v15. There is no runtime reward-version selector; historical experiment names may refer to older rewards.

### 8.3 Final-Approach Reward

Define

```
rho_ratio   = rho / max(0.1 s, 1e-6)
psi_ratio   = |psi_e| / 3 deg
speed_ratio = ||v_p|| / 0.15

center_quality    = exp(-3 rho_ratio^2)
near_field_weight = exp(-s / 50)

delta_rp  = previous(||r_p||) - current(||r_p||)
delta_rho = previous(rho) - current(rho).
```

The base reward is

```
r_final,base =
    - 0.03 ||r_p||
    - 3.0  ||v_p||
    - 0.5  rho
    - 5.0  |psi_e|
    - 1.0  |omega_z|
    - 0.02 ||u_norm||^2
    - 2.0  u_norm,alpha^2
    + 45.0 delta_rp (0.25 + 0.75 center_quality)
    + 5.0  delta_rho
    - 10.0 max(0, rho_ratio - 0.4)^2
    - 25.0 near_field_weight rho_ratio^2
    - 25.0 max(0, psi_ratio - 0.5)^2
    - 20.0 max(0, speed_ratio - 0.7)^2.
```

Angles and angular rates are represented internally in radians and radians per second. Degree values above define thresholds and normalization constants.

### 8.4 Event Terms

After the phase-specific base reward,

```
r_event =
    - 10   I_warn
    - 1000 I_fail
    + 1000 I_success
    - 200  I_timeout.
```

- `I_warn = 1` for each resulting warning state.
- `I_fail = 1` for a hard safety or numerical failure.
- `I_success = 1` for reaching the phase terminal condition.
- `I_timeout = 1` when the horizon is reached without an earlier failure or success.

## 9. Simplification Consequences and Limitations

1. The planar model is useful for developing phase logic, port-offset coupling, safety classification, and RL training workflows at lower computational cost.
2. Successful planar behavior does not demonstrate robustness to out-of-plane translation, roll/pitch error, full inertia coupling, or 3D approach geometry.
3. Direct acceleration commands assume an ideal lower-level controller and cannot predict thruster usage, fuel, minimum impulse, or allocation feasibility.
4. Omitting the IDSS-350T third-axis port offset changes the true 3D contact geometry.
5. The reduced orbital equations are not a substitute for a validated 6-DoF relative-orbit propagator.
6. The fixed target assumption excludes target attitude motion and target-port velocity.
7. The disturbance-free model cannot establish disturbance rejection.
8. The terminal tolerance is not a physical docking-contact or capture model.
9. Closing alignment is encouraged but is not a hard safety constraint.
10. Warning states are allowed and penalized; they are not equivalent to certified-safe nominal operation.
11. The closing `s > d_P2` classification gap is a known monitor limitation.
12. Results should be reported as results for the implemented planar environment, followed by validation in the general 6-DoF scenario during the design extension.

## 10. Implementation Authority

This document is intended to remain synchronized with:

- `core.py` for state propagation, geometry, initialization, safety, and termination;
- `gym_env.py` for observation construction, normalized actions, rewards, and Gymnasium termination/truncation semantics.

If either source file changes, the corresponding numerical or logical section here must be updated. The general scenario should change only when the intended 6-DoF mission specification itself changes.
