# run_rollout.py

import argparse, glob
from typing import List

from envs.twin import TwinEnv                     # swap to RealEnv later
from control.shield import ActionShield
from control.policy_random import RandomHeuristic, StaircaseHeuristic
from control.policy_cookbook import CookbookHeuristic, CookbookParams
from anneal_io.logger import StepLogger
from anneal_io.reward import step_reward, terminal_bonus

# Optional: IQL offline RL
try:
    from rl_iql.iql import train_iql, Policy as IQLPolicy
    HAS_IQL = True
except Exception:
    HAS_IQL = False


def build_policy(name: str, jsonl_paths: List[str], device: str, mode: str = "rt13"):
    name = name.lower()
    if name == "random":
        return RandomHeuristic(seed=0)
    if name == "stair":
        return StaircaseHeuristic()
    if name == "cookbook":
        return CookbookHeuristic(mode=mode, params=CookbookParams())
    if name == "iql":
        if not HAS_IQL:
            raise RuntimeError("IQL not available. Ensure rl_iql/ is installed and importable.")
        if not jsonl_paths:
            raise ValueError("IQL needs offline logs. Pass --logs 'runs/*.jsonl' or similar.")
        print(f"[IQL] Training on {len(jsonl_paths)} log files...")
        iql = train_iql(jsonl_paths, steps=100_000, batch_size=1024, device=device)
        return IQLPolicy(iql)  # exposes predict(obs)->Action
    raise ValueError(f"Unknown policy: {name}")


def run_episode(env, policy, shield, logger, goal_label="sqrt13", max_steps=12):
    obs = env.reset(goal_label)
    run_id = env.run_id
    obs_hist = [obs]
    done = False
    step_id = 0

    while not done and step_id < max_steps:
        step_id += 1

        # policy proposes an action
        a = policy.predict(obs)

        # safety clamp
        a, clamped = shield.clamp(a, obs)

        # execute one macro-segment (ramp + dwell)
        obs_next, _, done_flag, info = env.step(a)

        # compute offline reward (consistent across sim/real)
        r = step_reward(obs, obs_next, a)
        bonus, success = terminal_bonus(obs_hist + [obs_next])
        r_total = r + bonus

        # episode termination conditions
        done = done_flag or success

        # annotate info & log
        info.safety_clamped = clamped
        logger.write_row(run_id, step_id, goal_label, obs, a, obs_next, r_total, done, info)

        # advance
        obs = obs_next
        obs_hist.append(obs_next)

    return run_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, default="random",
                        choices=["random", "stair", "cookbook", "iql"],
                        help="Which controller to use.")
    parser.add_argument("--mode", type=str, default="rt13",
                        choices=["rt13", "htr"],
                        help="Cookbook mode (only used if --policy cookbook).")
    parser.add_argument("--logs", type=str, default="",
                        help="Glob for JSONL logs to train IQL, e.g. 'runs/*.jsonl'")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=12)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--goal", type=str, default="sqrt13")
    args = parser.parse_args()

    # Build env, shield, logger
    env = TwinEnv(seed=42)               # later: RealEnv(...)
    shield = ActionShield()
    logger = StepLogger()

    # Resolve logs for IQL (if any)
    jsonl_paths = []
    if args.logs:
        jsonl_paths = sorted(glob.glob(args.logs))
        if not jsonl_paths:
            print(f"[warn] No logs matched pattern: {args.logs}")

    # Build policy
    policy = build_policy(args.policy, jsonl_paths, device=args.device, mode=args.mode)

    # Roll episodes
    for ep in range(args.episodes):
        run_id = run_episode(env, policy, shield, logger,
                             goal_label=args.goal, max_steps=args.max_steps)
        print(f"episode {ep+1}/{args.episodes} → run_id={run_id}")

    env.close()


if __name__ == "__main__":
    main()
