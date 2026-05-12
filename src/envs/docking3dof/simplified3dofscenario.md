# Simplified 3-DoF Scenario Setup

## 1. Scope and Relationship to General Scenario

This document defines the 3-DoF planar variant of the docking problem.

- All definitions from the general scenario remain valid unless explicitly overridden here.
- This file focuses on additional, modified, or specialized concepts for the planar model.

## 2. Modeled Problem (3-DoF Interpretation)

The mission is still split into two sequential control tasks:

1. Closing phase (P2 -> P3)
2. Final approach phase (P3 -> Docking)

In the simplified model, relative motion is represented by:

- Two translational DoFs in the target-centered docking plane (`x_O`, `y_O`)
- One rotational DoF: relative in-plane angle about out-of-plane axis (`z_O`)

Not modeled in this variant:

- Out-of-plane translation
- Full 3D attitude dynamics
- Angular rates about axes other than `z_O`
- Post-contact capture/latching dynamics

Motivation: This is a practical intermediate model between pure translation and full 6-DoF docking, allowing interface-alignment effects to be represented with moderate complexity.

### 2.1 Controller 1: Corridor Capture (P2 -> P3)

Controller 1 drives the chaser from close-range point P2 to corridor-entry point P3 under planar and rotational constraints.

Objectives:

- Reduce relative range and stabilize near P3.
- Reduce planar relative velocity.
- Prepare planar approach geometry near nominal docking axis.
- Reduce attitude error and angular rate for terminal-phase handover.

### 2.2 Controller 2: Terminal Planar Docking (P3 -> Docking)

Controller 2 performs the final in-plane corridor approach and contact alignment.

Objectives:

- Enter and follow the planar docking corridor.
- Keep lateral deviation small.
- Continue reducing closing speed near contact.
- Achieve admissible planar attitude alignment.
- Satisfy terminal translational and rotational docking conditions.

## 3. Planar Docking-Port Geometry and Kinematics

Relative-motion frame is centered at target CoM and restricted to the docking plane.

Let:

- `r = [x, y]^T`: chaser CoM relative position in target planar frame
- `psi`: relative in-plane orientation of chaser
- `R(psi)`: planar rotation mapping chaser-body vectors into target planar frame
- `r_c = [r_cx, r_cy]^T`: chaser docking-port center offset from chaser CoM (body frame)
- `r_t = [r_tx, r_ty]^T`: target docking-port center offset from target CoM (target planar frame)

Docking-port relative position:

```
r_p = r + R(psi) * r_c - r_t
```

Planar rotation matrix:

```
R(psi) = [[cos(psi), -sin(psi)], [sin(psi), cos(psi)]]
```

Docking contact condition:

```
r_p = 0
```

If `v = [x_dot, y_dot]^T` and `omega = psi_dot`, docking-port relative velocity is:

```
v_p = v + omega * S * R(psi) * r_c
```

with planar skew-symmetric matrix:

```
S = [[0, -1], [1, 0]]
```

This allows terminal constraints to be imposed at docking-port level on `r_p`, `v_p`, `psi`, and `omega`.

## 4. Formal 3-DoF Problem Definition

### 4.1 State

```
x(t) = [r(t), v(t), psi(t), omega(t)]
```

Where:

- `r(t) in R^2`: relative position in target plane (`x_O`, `y_O`)
- `v(t) in R^2`: relative velocity in target plane
- `psi(t)`: relative in-plane orientation angle (about `z_O`)
- `omega(t)`: scalar angular velocity (about `z_O`)

### 4.2 Control Input

```
u(t) = [a_c(t), alpha_c(t)]
```

Where:

- `a_c(t) in R^2`: commanded translational acceleration of the chaser in the x and y directions of the target-centered orbital frame
- `alpha_c(t) in R`: commanded angular acceleration about the z-axis

### 4.3 Admissible Control Set

```
U = { u :
    ||a_c|| <= a_max AND
    ||alpha_c|| <= alpha_max
}
```
a_x_max = 0.01 m/s^2
a_y_max = 0.01 m/s^2
alpha_z_max = 0.003 rad/s^2

## 5. 3-DoF Derived Quantities and Sets

### 5.1 Docking Quantities

Docking-port relative position:

```
r_p = r + R(psi) * r_c - r_t
```

Docking axis (planar projection):

```
e_d = [1, 0]
```

Longitudinal distance:

```
s = dot(e_d, r_p)
```

Lateral offset:

```
rho = | y-component of r_p |
```

### 5.2 Primitive Geometric Sets (Planar Versions)

Approach ellipsoid becomes 2D ellipse:

```
E_AE = { r : r^T Q_AE r <= 1 }
Q_AE = diag(1/2000^2, 1/1000^2)
```

Keep-out sphere becomes circle:

```
S_KOS = { r : ||r|| <= 200 }
```

Docking cone becomes 2D wedge:

```
C_dock = { x :
    0 <= s <= 2000 AND
    rho <= (200 / 2000) * s
}
```

Docking corridor becomes strip:

```
C_corr = { x :
    0 <= s <= d_P3 AND
    rho <= 5
}
```

### 5.3 Alignment Sets (Planar)

```
A_safe = {
    |psi_e| <= 3 deg AND
    |omega| <= 0.15 deg/s
}

A_warn = {
    (3 < |psi_e| <= 4) OR
    (0.15 < |omega| <= 0.2)
}

A_unsafe = {
    |psi_e| > 4 OR
    |omega| > 0.2
}
```

### 5.4 Velocity Sets

Closing phase:

```
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
```

Final approach phase:

```
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
```

### 5.5 Terminal Docking Set

```
X_dock = {
    ||r_p|| <= 0.1 AND
    ||v_p|| <= 0.1 AND
    |psi_e| <= 4 deg AND
    |omega| <= 0.2 deg/s
}
```

### 5.6 Initialization Sets

```
X_P2 = {
    s in [d_P2_min, d_P2_max],
    rho <= rho_P2,
    v_parallel in [v_min_P2, v_max_P2],
    |v_perp| <= v_perp_P2,
    |psi_e| <= psi_P2,
    |omega| <= w_P2
}

X_P3 = {
    s in [d_P3_min, d_P3_max],
    rho <= rho_P3,
    v_parallel in [v_min_P3, v_max_P3],
    |v_perp| <= v_perp_P3,
    |psi_e| <= psi_P3,
    |omega| <= w_P3
}
```

## 6. Safety and Phase Logic

### 6.1 Safety Sets

Closing phase geometry layer:

```
G_safe_close = (E_AE \ S_KOS) ∩ C_dock   with   d_P3 <= s <= d_P2
G_unsafe_close = S_KOS \ C_dock
G_warn_close = (E_AE \ (S_KOS ∪ C_dock)) ∪ (S_KOS ∩ C_dock)
G_out_close = E_AE_complement
```

Closing combined sets:

```
X_safe_close = G_safe_close ∩ V_safe_close
X_unsafe_close = G_unsafe_close ∪ V_unsafe_close
X_warn_close = (G_warn_close ∪ V_warn_close) \ X_unsafe_close
```

Final-approach geometry layer:

```
G_safe_final = C_dock ∩ C_corr
G_unsafe_final = C_dock_complement
G_warn_final = C_dock \ C_corr
```

Final-approach combined sets:

```
X_safe_final = G_safe_final ∩ V_safe_final ∩ A_safe
X_unsafe_final = G_unsafe_final ∪ V_unsafe_final ∪ A_unsafe
X_warn_final = (G_warn_final ∪ V_warn_final ∪ A_warn) \ X_unsafe_final
```

### 6.2 Phase Definitions

Closing phase (P2 -> P3):

- Initial: `x(t_P2) in X_P2`
- During: `x(t) in X_safe_close`, `u(t) in U`
- Final: `x(t_P3) in X_P3`

Final approach phase (P3 -> Docking):

- Initial: `x(t_P3) in X_P3`
- During: `x(t) in X_safe_final`, `u(t) in U`
- Final: `x(t_f) in X_dock`

## 7. 3-DoF Assumptions Register

Use each item as:

- Assumption
- Consequence:
- Limitation:

1. The model includes in-plane relative translation and one in-plane rotational DoF.
   Consequence:
   Limitation:

2. Docking motion is planar in the target-attached `x_T-y_T` plane.
   Consequence:
   Limitation:

3. Out-of-plane position and out-of-plane velocity are assumed zero in this scenario.
   Consequence:
   Limitation:
