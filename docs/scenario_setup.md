### PROBLEM SETUP ###

## Modeled problem

The modeled control problem includes the closing phase and the final approach phase of close-proximity rendezvous. These phases are treated as one continuous autonomous control task.

![Docking Phases](docking_phases.png) (Luo, J. Zhang, and Tang, 2014)

### Closing phase

The chaser reduces the relative distance further and transfers from the previous near-target point (a couple of km away from the target), P2, to a station-keeping point, P3 (hundreds of meters from the target). Therefore, the controller is required to:

- **reduce range to the target and stabilize at or near the handover point**: move the chaser from the initial close-range state, P2, to a terminal point or corridor-entry point, P3, located hundreds of meters away from the target
- **reduce relative velocity**: slow the relative motion down as the range decreases
- **prepare geometry for final approach**: bring the chaser into the correct side or region so that it can enter the final straight-line approach corridor

### Final approach phase

Starting from P3, the chaser approaches the target along a straight line as much as possible and must satisfy strict requirements on relative position, velocity, attitude, and angular rate at contact. Therefore, the controller is required to:

- **enter the approach gate or corridor and track a straight-line approach**: once at or near P3, move into the permitted line of approach and continue along a straight line
- **keep lateral error small**: stay centered with respect to the docking point and the line-of-sight (LOS) cone
- **continue reducing approach speed**: ensure that the closing speed becomes smaller as contact approaches
- **arrive with admissible terminal conditions**: satisfy the required constraints on relative position, velocity, attitude, and angular rate at contact with docking point (center of target docking interface)

### Docking geometry 

The **docking axis** is defined as the centerline of the target docking port, i.e. the axis normal to the target docking interface plane and pointing outward from the target port. In the present model, this axis is denoted by `e_d` and is assumed to coincide with one axis of the target-centered frame. The docking corridor is aligned with this axis, which therefore represents the nominal approach direction of the chaser.

The **target docking-port center**, expressed relative to the target center of mass, is defined as a fixed offset along the docking axis and is assumed to be located on the front face of the target body.

The **chaser docking-port center**, expressed relative to the chaser center of mass, is defined as a fixed offset along the chaser body `x`-axis and is assumed to be located on the front face of the chaser body.

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

Assumption: The closing phase and the final approach phase are modeled as one continuous autonomous control task, without introducing a separate control formulation at the transition between the two phases.
Consequence: 
Limitation:

Assumption: docking interface centerline corresponds with the spacecraft centerline 
Consequence: 
Limitation:


