# Contributing and Git Workflow

## Branching strategy

The `main` branch stores the stable version of the project. New work should be developed in separate feature branches created from `main`.

Branch names should follow these conventions:

- `feature/<task-name>` for new functionality
- `fix/<issue-name>` for bug fixes
- `docs/<topic>` for documentation updates
- `testing/<topic>` for experimental work on study cases

Examples:
- `feature/project-structure`
- `feature/3dof-simulator`
- `feature/lqr-baseline`
- `fix/reset-function-found-bug`
- `docs/readme-update`

## Development workflow

1. Start from the latest version of `main`.
2. Create a new branch for the task.
3. Implement and test the changes in that branch.
4. Commit changes with clear and concise commit messages.
5. Open a pull request to merge the branch into `main`.
6. Merge only after the changes are reviewed and the branch is in a stable condition.

## Pull requests

Pull requests are used to keep a clear history of development versions and completed tasks. Each pull request should correspond to one main task or one logically grouped set of changes.

Pull request titles should clearly describe the implemented change.

Examples:
- `Add initial project folder structure`
- `Implement 3-DoF simulator skeleton`
- `Add LQR baseline controller`

## Commit message conventions

Commit messages should be short, clear, and action-oriented.

Examples:
- `Create initial repository structure`
- `Add simulator reset function`
- `Implement state propagation`
- `Update README with setup instructions`

## General rules

- Keep `main` as the stable branch.
- Do not commit incomplete experimental code directly to `main`.
- Use pull requests to document the development history.
- Keep branches focused on one task whenever possible.

## Environment 
This project is developed using Python 3.12.2.