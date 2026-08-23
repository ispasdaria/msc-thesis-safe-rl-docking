# Scientific-article draft: LQR baseline results

## A. LQR baseline performance

Table 2 summarizes the performance of the selected phase-specific LQR controllers on the nonlinear simulator. In the isolated closing-phase evaluation, the controller reached the sampled $P_3$ handover set in 41 of 50 rollouts, corresponding to a success rate of 82.0% (95% Wilson confidence interval: 69.2--90.2%). The remaining nine cases reached the 12,000 s limit; none terminated because of a hard safety violation. Across all 50 rollouts, the mean absolute terminal errors were 19.50 m and 1.38 m in axial and lateral position, 0.0072 m/s and 0.0045 m/s in axial and lateral velocity, $0.255^\circ$ in attitude, and $0.0242^\circ$/s in angular rate. Position was the most frequent unmet handover requirement: the axial and lateral tolerances were exceeded in 10% and 12% of the cases, respectively. The mean elapsed time was 11,236.7 s, or 93.6% of the available phase duration.

The closing controller requested at least one input outside its admissible bound during 99.5% of the simulation steps. This high clipping fraction did not produce a hard-safety termination, but it shows that the long-range maneuver is governed largely by the actuator limits rather than by unconstrained LQR behavior. The closing result should therefore be interpreted as hard-safety-preserving in the observed sample but convergence-limited: some initial-condition and target combinations cannot be brought into the complete six-component $P_3$ tolerance set before the time limit. This is the main limitation of the LQR baseline and identifies the closing phase, rather than terminal docking, as the dominant source of mission failure.

The isolated final-approach controller completed all 50 rollouts (95% Wilson confidence interval: 92.9--100%) without a timeout or hard-safety violation. The mean docking-port distance and speed at termination were 0.097 m and 0.005 m/s, respectively, while the mean attitude and angular-rate errors were numerically negligible. The mean completion time was 3,750.0 s and at least one control channel was clipped during only 0.1% of the steps. These results show that formulating the feedback error at the displaced docking port, while retaining a constant LQR gain, provides accurate interface regulation under the nonlinear propagation model.

The end-to-end evaluation produced 235 successful dockings in 300 missions, giving an overall success rate of 78.3% (95% Wilson confidence interval: 73.3--82.6%). All 65 unsuccessful missions timed out during closing. Every mission that reached $P_3$ subsequently completed final approach, yielding a conditional final-approach success rate of 235/235, or 100% (95% Wilson confidence interval: 98.4--100%). No hard-safety termination occurred in either phase; for zero events in 300 missions, the upper bound of the 95% Wilson interval is 1.26%. Hence, the phase transition introduced no observed loss of reliability: the state delivered by the closing controller was compatible with the port-error final controller in every attempted docking. The difference between isolated closing success (82.0%) and complete-mission success (78.3%) reflects the larger set of closing initializations, not additional failures after handover.

The matched nominal comparisons in Figs. 2 and 3 isolate the effect of the proposed distance-dependent velocity references. With a fixed reference equal to the sampled $P_3$ axial velocity, the closing controller timed out at 12,000 s with 302.0 m of axial error. Using the scheduled reference with the same gain and initial condition reduced the error to the 20 m handover tolerance and completed the phase in 10,968 s. Similarly, a zero axial port-velocity reference caused final approach to time out after 5,000 s at a port distance of 99.48 m, whereas the scheduled reference completed docking in 3,742 s at a distance of 0.096 m and a speed of 0.005 m/s. Thus, the main LQR contribution is not a different Riccati solution alone, but the combination of physically normalized phase-specific costs, docking-port-aware feedback coordinates, and distance-dependent reference shaping. Together, these elements turn constant-gain LQR controllers into a reproducible, safety-preserving baseline for the nonlinear two-phase docking problem, while the closing saturation and timeout rates define a clear benchmark for the subsequent learning-based controller.

## Table 2. Monte Carlo performance of the selected LQR controllers

| Evaluation | Runs | Successful outcomes | Timeouts | Hard-safety terminations | Mean elapsed time [s] | Mean any-channel clipping fraction | Terminal result |
|---|---:|---:|---:|---:|---:|---:|---|
| Closing phase, seeds 0--49 | 50 | 41 (82.0%) | 9 (18.0%) | 0 (0%) | 11,236.7 | 99.5% | Mean $|e_x|=19.50$ m, $|e_y|=1.38$ m, $|e_{v_x}|=0.0072$ m/s, $|e_{v_y}|=0.0045$ m/s |
| Final approach, seeds 0--49 | 50 | 50 (100%) | 0 (0%) | 0 (0%) | 3,750.0 | 0.1% | Mean $\lVert r_p\rVert=0.097$ m and $\lVert v_p\rVert=0.005$ m/s |
| Complete mission, seeds 0--299 | 300 | 235 (78.3%) | 65 (21.7%), all during closing | 0 (0%) | -- | -- | Final approach succeeded in 235/235 attempts (100%) |

The phase-level terminal quantities are means over all 50 terminations, including timeouts. "Any-channel clipping" is the fraction of steps at which at least one raw body-frame command exceeded its bound; it does not mean that all three channels were saturated simultaneously. The confidence intervals quoted in the text are two-sided 95% Wilson intervals.

## Recommended figures and captions

### Figure 2: closing-reference ablation

Use [closing_velocity_schedule_comparison.pdf](../src/controllers/LQR-3DoF/results/development_comparisons/closing_velocity_schedule_comparison.pdf).

**Caption:** Matched nominal closing-phase comparison on the nonlinear simulator. The controller gain, initial condition, sampled $P_3$ target, actuator limits, and propagation settings are identical; only the axial-velocity reference is changed. The fixed $P_3$-velocity reference times out at 12,000 s with 302.0 m of axial error, whereas the distance-scheduled reference reaches the 20 m axial handover tolerance in 10,968 s. The faint traces in (b) are the unsmoothed simulator velocities and the solid traces are 60 s moving averages.

### Figure 3: final-reference ablation

Use [final_velocity_schedule_comparison.pdf](../src/controllers/LQR-3DoF/results/development_comparisons/final_velocity_schedule_comparison.pdf).

**Caption:** Matched nominal final-approach comparison on the nonlinear simulator. With the same port-error LQR gain, a zero axial port-velocity reference stalls and times out at 5,000 s with a 99.48 m port-distance error. The distance-scheduled reference preserves approach progress and reduces the velocity near contact, reaching the 0.1 m docking set in 3,742 s with a port speed of 0.005 m/s. The solid velocity traces in (b) are 60 s moving averages.

## Supporting diagnostic figures

The following figures were generated from the saved median successful complete mission (seed 152). They are useful for the thesis or supplementary material, but the schedule ablations above provide a stronger contribution-focused use of the article's limited figure space:

- [complete-mission overview](scientific_article_lqr_results_figures/seed0152_median_full_success_success_overview.png)
- [complete-mission actions](scientific_article_lqr_results_figures/seed0152_median_full_success_success_actions.png)
- [complete-mission phase planes](scientific_article_lqr_results_figures/seed0152_median_full_success_success_phase_plane.png)

## Author notes before submission

1. The 50-rollout phase results use the same seeds employed during candidate tuning, so they are tuning-set rather than held-out results. The 300-mission batch is stronger end-to-end evidence, but seeds 0--49 still overlap with tuning. A final non-overlapping validation batch (for example, seeds 300--599) would support a stronger generalization claim.
2. The schedule figures are controlled single-case ablations. Describe them as mechanistic evidence for why scheduling is necessary, not as statistical evidence of an average improvement.
3. The closing raw commands substantially exceed the input bounds. Avoid calling the implemented closing response an unconstrained LQR optimum; it is a saturated LQR tracking controller operating on the nonlinear plant.
4. If one additional article figure fits, use a compact end-to-end trajectory/safety-envelope plot. The current six-panel overview is better suited to the thesis because its diagnostic detail will be difficult to read in a two-column article.
