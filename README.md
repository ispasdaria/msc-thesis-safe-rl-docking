# Safe Reinforcement Learning for Autonomous Close-Proximity Spacecraft Docking

## Project description

This repository contains the implementation developed for my MSc thesis on safe reinforcement learning for autonomous close-proximity spacecraft docking.

The project investigates a hybrid control architecture that combines:
- a classical LQR baseline controller,
- a Soft Actor-Critic (SAC) controller,
- and a Control Barrier Function (CBF)-based safety filter.

The objective is to evaluate whether the inclusion of a model-based safety filter can improve safety-constraint satisfaction during training and evaluation, while maintaining useful docking performance in simulation.

## Scope and planned components

The implementation is planned to develop progressively through the following main components:

- a simplified 3-DoF close-proximity docking simulation environment,
- an 6-DoF extension to include rotational dynamics,
- an LQR baseline controller,
- a Soft Actor-Critic controller,
- a CBF-based safety filter,
- and an evaluation framework for comparing the controller configurations.

The main comparison cases are expected to include the LQR baseline, SAC trained from scratch, warm-started SAC if feasible, SAC combined with the CBF filter, and possibly warm-started SAC combined with the CBF filter.
