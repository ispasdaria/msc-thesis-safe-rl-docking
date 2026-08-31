# 7.1 LQR baseline performance


## 7.1.4 Held-out complete-mission evaluation

After phase-level tuning, both controllers and their velocity schedules were
frozen and evaluated together over 600 held-out complete missions using seeds
300--899. The propagated terminal state of the closing phase was passed
directly to the final-approach environment; no new \(P_3\) state was sampled at
the phase transition. Final approach was attempted only when closing completed
successfully.

Of the 600 missions, 490 completed docking successfully. This corresponds to
an observed complete-mission success rate of 81.7% and a 95% Wilson interval
of 78.4--84.6%. All 110 unsuccessful missions terminated during closing:
109 reached the closing time limit and one encountered a hard velocity-safety
violation. In the latter case, seed 601, the docking-port-relative speed
reached 0.3519 m/s, marginally exceeding the 0.35 m/s hard limit. The observed
complete-mission hard-safety-termination rate was therefore 1/600, or 0.17%,
with a 95% Wilson interval of 0.03--0.94%. The lower and upper limits show that
one observed event provides only a relatively imprecise estimate of this rare
outcome.

Every mission that reached the \(P_3\) handover set subsequently docked. The
conditional final-approach rate was therefore 490/490, or 100%, with a 95%
Wilson interval of 99.2--100%. Since final approach was attempted only after
successful closing, the mission success probability can be decomposed as

\[
P(M)=P(C)P(F\mid C),
\]

where \(C\) denotes successful closing and \(F\) denotes successful final
approach. For the held-out sample,

\[
\hat P(M)=\frac{490}{600}\frac{490}{490}=\frac{490}{600}.
\]

This decomposition confirms that the final-approach transition introduced no
additional observed failure in the held-out campaign. End-to-end reliability
was governed by whether the closing controller reached the handover set.

**Table 7.4. Held-out full-mission LQR evaluation.**

| Metric | Result |
|---|---:|
| Evaluation seeds | 300--899 |
| Number of missions | 600 |
| Successful missions | 490 (81.7%) |
| 95% Wilson interval for mission success | 78.4--84.6% |
| Closing timeouts | 109 (18.2%) |
| Hard-safety terminations | 1 (0.17%) |
| 95% Wilson interval for safety termination | 0.03--0.94% |
| Phase of all unsuccessful outcomes | Closing |
| Final-approach attempts | 490 |
| Successful final-approach attempts | 490/490 (100%) |
| 95% Wilson interval for conditional final success | 99.2--100% |

*Insert representative full-mission figure here. Suggested caption: Exact
docking-port-relative position, velocity, attitude, and angular-rate histories
for a representative held-out successful mission. The vertical line denotes
the \(P_3\) controller transition and the shaded regions distinguish closing
from final approach.*

*Insert Monte Carlo convergence figure here. Suggested caption: Cumulative
held-out mission-success estimate as a function of rollout count, with 95%
Wilson limits and the 5th--95th percentile range obtained from repeated random
permutations of the 600 mission records.*

## 7.1.5 Overall interpretation

The LQR baseline demonstrates that the modeled docking task can be completed
reliably once an admissible \(P_3\) handover state is reached. The
final-approach controller succeeded for every isolated and end-to-end attempt
and required little actuator clipping or accumulated control effort. The
dominant limitation lies in closing. Although the closing controller remained
inside the hard safety set in nearly every held-out mission, it frequently
required the entire available duration and did not always satisfy the six
handover conditions simultaneously.

The closing results also demonstrate why success percentage alone is
insufficient for characterizing a baseline. The controller achieved an
approximately 82% success rate, but did so with persistent translational
clipping, a mean \(\Delta v\)-type proxy of 158.708 m/s, and one-step command
alternation in the examined representative history. These properties are
important when the LQR baseline is compared with learned or safety-filtered
controllers: a controller should be assessed not only by whether it docks,
but also by the distribution of failure modes, duration, safety events,
actuator use, and accumulated applied-control magnitude.

The reported probabilities remain conditional on the modeled dynamics,
initial-condition distributions, actuator representation, and termination
logic. In particular, the absence of spacecraft mass, thruster allocation,
minimum impulse effects, navigation error, and external disturbances means
that neither the confidence intervals nor the control-effort proxies constitute
flight qualification or a direct propellant prediction. They provide a
reproducible simulation-level reference against which the subsequent
reinforcement-learning and safety-filtered controllers can be evaluated.
