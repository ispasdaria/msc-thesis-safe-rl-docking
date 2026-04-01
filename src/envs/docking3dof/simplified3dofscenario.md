### PROBLEM SETUP ###

## Modeled problem

For the 3-DoF translational model, the modeled control problem includes the closing phase and the final approach phase of close-proximity rendezvous. These phases are treated as one continuous autonomous control task.

In this simplified model, only the relative translational motion of the chaser with respect to the target is considered. Relative attitude dynamics, angular rates, and post-contact mechanical docking effects are not modeled explicitly. The control objective is therefore to guide the chaser safely from the initial close-range region to first contact with admissible translational terminal conditions.

### Closing phase

The chaser reduces the relative distance further and transfers from the previous near-target point, P2 (a couple of kilometers away from the target), to a station-keeping point, P3 (hundreds of meters from the target). Therefore, the controller is required to:

- **reduce range to the target and stabilize at or near the handover point**: move the chaser from the initial close-range state, P2, to a terminal point or corridor-entry point, P3, located hundreds of meters away from the target
- **reduce relative velocity**: slow the relative translational motion down as the range decreases
- **prepare geometry for final approach**: bring the chaser into the correct side or region so that it can enter the final straight-line approach corridor

### Final approach phase

Starting from P3, the chaser approaches the target along a straight line as much as possible and must satisfy strict requirements on relative position and relative velocity at contact. Therefore, the controller is required to:

- **enter the approach gate or corridor and track a straight-line approach**: once at or near P3, move into the permitted line of approach and continue along a straight line
- **keep lateral error small**: stay centered with respect to the docking point and the line-of-sight (LOS) cone
- **continue reducing approach speed**: ensure that the closing speed becomes smaller as contact approaches
- **arrive with admissible translational terminal conditions**: satisfy the required constraints on relative position and relative velocity at first contact with the docking point, taken here as the center of the target docking interface

### 3-DoF modeling interpretation

Within the 3-DoF translational formulation, the state captures only relative position and relative velocity, and the controller commands translational motion. In this sense, successful completion of the modeled task means that the chaser reaches first contact with acceptable translational alignment and closing conditions, while rotational alignment and capture mechanics are left outside the scope of this model.