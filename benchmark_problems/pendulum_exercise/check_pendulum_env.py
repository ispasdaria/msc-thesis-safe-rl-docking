"""Run a smoke check of Gymnasium's Pendulum environment."""

from argparse import ArgumentParser

import gymnasium as gym


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--render",
        action="store_true",
        help="open a window and render the smoke check",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="number of random-action steps to run (default: 300)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    render_mode = "human" if args.render else None
    env = gym.make("Pendulum-v1", render_mode=render_mode)
    try:
        observation, _ = env.reset()

        for _ in range(args.steps):
            action = env.action_space.sample()
            observation, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                observation, _ = env.reset()
    finally:
        env.close()


if __name__ == "__main__":
    main()
