"""CPU PPO local navigator: imitation initialization, then reward-based PPO.

python -B experiments/29_train_navigation.py --seeds 0 1 2 --timesteps 120000
All budgets are total transitions across vector environments. Training emits
BC, final-PPO and best positive-PPO-step checkpoints separately.
"""

import argparse
from dataclasses import asdict
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

from src.navigation.observations import OBSERVATION_VERSION, teacher_action
from src.navigation.rl_env import NavigationEnv


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def evaluate(model, env, seeds):
    results = []
    started = time.perf_counter()
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        reward_sum = 0.
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            reward_sum += reward
            if terminated or truncated:
                results.append({"seed": seed, "success": info["is_success"], "collision": info["collision"],
                                "steps": info["steps"], "final_error_m": info["distance"],
                                "return": reward_sum, "truncated": truncated})
                break
    return {"episodes": len(results), "success_rate": sum(r["success"] for r in results) / len(results),
            "collision_rate": sum(r["collision"] for r in results) / len(results),
            "mean_steps": float(np.mean([r["steps"] for r in results])),
            "mean_return": float(np.mean([r["return"] for r in results])),
            "wall_seconds": time.perf_counter() - started, "cases": results}


def imitation_initialize(model, env, seed, samples=24000, epochs=60):
    rng = np.random.default_rng(seed + 500)
    inputs, targets = [], []
    started = time.perf_counter()
    while len(inputs) < samples:
        env.reset(seed=int(rng.integers(1, 2 ** 31)))
        start = env.pose[:2].copy()
        target = np.array(env.target)
        for _ in range(24):
            fraction = rng.uniform(0., 1.)
            if rng.random() < .45:
                # Dense samples around arrival and intermediate waypoint disks.
                distance = max(math.dist(start, target), .001)
                fraction = max(0., 1. - rng.uniform(.001, .35) / distance)
            env.pose[:2] = start * (1 - fraction) + target * fraction
            env.pose[2] = rng.uniform(-math.pi, math.pi)
            env.motion[:] = [rng.uniform(0., .3), rng.uniform(-.3, .3)]
            env.final = bool(rng.random() < .7)
            observation = env.observation()
            inputs.append(observation)
            targets.append(teacher_action(observation))
            if len(inputs) >= samples:
                break
    x, y = torch.tensor(np.asarray(inputs)), torch.tensor(np.asarray(targets))
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=.001)
    losses = []
    for epoch in range(epochs):
        order = rng.permutation(samples)
        total = 0.
        for offset in range(0, samples, 256):
            index = order[offset:offset + 256]
            mean = model.policy.get_distribution(x[index]).distribution.mean
            loss = torch.mean((mean - y[index]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(index)
        losses.append(total / samples)
        if (epoch + 1) % 10 == 0:
            print(f"BC epoch {epoch + 1}/{epochs}: MSE {losses[-1]:.5f}", flush=True)
    with torch.no_grad():
        model.policy.log_std.fill_(-3.)
    return {"samples": samples, "epochs": epochs, "losses": losses,
            "wall_seconds": time.perf_counter() - started, "teacher": "smooth local feedback controller",
            "note": "supervised initialization only; separate from PPO environment interactions"}


class ValidationCallback(BaseCallback):
    def __init__(self, env, destination, interval=20000, episodes=60):
        super().__init__()
        self.env, self.destination, self.interval = env, destination, interval
        self.next_eval = interval
        self.best_score = -math.inf
        self.best_step = None
        self.records = []
        self.episodes = episodes

    def _on_step(self):
        if self.num_timesteps < self.next_eval:
            return True
        self.next_eval += self.interval
        report = evaluate(self.model, self.env, range(9000, 9000 + self.episodes))
        score = report["success_rate"] - report["collision_rate"] - report["mean_steps"] / 100000.
        report["ppo_steps"] = self.num_timesteps
        if score > self.best_score:
            self.best_score, self.best_step = score, self.num_timesteps
            self.model.save(str(self.destination / "best.zip"))
        self.records.append(report)
        write_json(self.destination / "validation.json", {"records": self.records, "best_step": self.best_step})
        print(f"PPO step {self.num_timesteps}: validation success {report['success_rate']:.0%}, collision {report['collision_rate']:.0%}, mean steps {report['mean_steps']:.1f}", flush=True)
        return True


def train_one(args, seed):
    destination = args.output / f"seed_{seed}"
    destination.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    env = NavigationEnv()
    check_env(env, warn=True)
    vec = DummyVecEnv([lambda: NavigationEnv(planner=env.planner) for _ in range(args.envs)])
    model = PPO("MlpPolicy", vec, seed=seed, device="cpu", learning_rate=args.learning_rate,
                n_steps=1024, batch_size=256, n_epochs=5, gamma=.995, gae_lambda=.95,
                clip_range=.15, target_kl=.02, ent_coef=0.,
                policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}}, verbose=0)
    started = time.perf_counter()
    if args.initial_policy is None:
        warmup = imitation_initialize(model, env, seed, samples=args.bc_samples, epochs=args.bc_epochs)
    else:
        initial = PPO.load(str(args.initial_policy), device="cpu")
        if initial.observation_space.shape != (25,) or initial.action_space.shape != (2,):
            raise ValueError("initial policy has an incompatible observation/action contract")
        model.policy.load_state_dict(initial.policy.state_dict())
        warmup = {"source": str(args.initial_policy), "source_sha256": hashlib.sha256(args.initial_policy.read_bytes()).hexdigest(),
                  "samples": 0, "epochs": 0, "note": "actor/critic weights initialized from supplied checkpoint; new PPO optimizer and step counter"}
    model.save(str(destination / "imitation.zip"))
    warmup_validation = evaluate(model, NavigationEnv(planner=env.planner, randomize=False), range(9000, 9000 + args.validation_episodes))
    print(f"Seed {seed}: imitation validation success {warmup_validation['success_rate']:.0%}", flush=True)
    callback = ValidationCallback(NavigationEnv(planner=env.planner), destination,
                                  interval=min(20000, args.timesteps), episodes=args.validation_episodes)
    learning_started = time.perf_counter()
    model.learn(total_timesteps=args.timesteps, callback=callback)
    learning_wall = time.perf_counter() - learning_started
    model.save(str(destination / "final.zip"))
    if callback.best_step is None:
        raise RuntimeError("training produced no positive-step PPO validation checkpoint")
    best = PPO.load(str(destination / "best.zip"), device="cpu")
    held_out = evaluate(best, NavigationEnv(planner=env.planner), range(args.test_start, args.test_start + args.test_episodes))
    manifest = {"seed": seed, "algorithm": "PPO continuous local control, imitation initialized",
                "observation_version": OBSERVATION_VERSION, "observation_size": 25,
                "navigation_config": asdict(env.planner.config), "envs": args.envs, "threads": args.threads,
                "device": "cpu", "network": {"actor": [64, 64], "critic": [64, 64]},
                "hyperparameters": {"learning_rate": args.learning_rate, "n_steps": 1024, "batch_size": 256,
                                    "n_epochs": 5, "gamma": .995, "gae_lambda": .95, "clip_range": .15},
                "requested_ppo_steps": args.timesteps, "actual_ppo_steps": model.num_timesteps,
                "best_ppo_step": callback.best_step, "imitation": warmup,
                "imitation_validation": warmup_validation, "held_out": held_out,
                "ppo_learning_wall_seconds": learning_wall,
                "ppo_transitions_per_second_including_validation": model.num_timesteps / learning_wall,
                "total_wall_seconds": time.perf_counter() - started,
                "versions": {name: metadata.version(name) for name in ("numpy", "torch", "gymnasium", "stable-baselines3")},
                "best_checkpoint_sha256": hashlib.sha256((destination / "best.zip").read_bytes()).hexdigest(),
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (Path(__file__), ROOT / "src/navigation/rl_env.py",
                                            ROOT / "src/navigation/observations.py", ROOT / "src/navigation/ppo_navigator.py")},
                "training_data_scope": "random locally visible goals on known campus, no simulator litter labels or images",
                "validation_episode_seeds": [9000, 9000 + args.validation_episodes - 1],
                "held_out_episode_seeds": [args.test_start, args.test_start + args.test_episodes - 1]}
    write_json(destination / "manifest.json", manifest)
    vec.close()
    print(f"Seed {seed}: final held-out success {held_out['success_rate']:.0%}; training wall {manifest['total_wall_seconds']:.1f} s", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--timesteps", type=int, default=1000000)
    parser.add_argument("--envs", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--bc-samples", type=int, default=24000)
    parser.add_argument("--bc-epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--output", type=Path, default=ROOT / "models/navigation/ppo_full")
    parser.add_argument("--initial-policy", type=Path)
    parser.add_argument("--validation-episodes", type=int, default=60)
    parser.add_argument("--test-start", type=int, default=25000)
    parser.add_argument("--test-episodes", type=int, default=100)
    args = parser.parse_args()
    if any(v < 1 for v in (args.timesteps, args.envs, args.threads, args.bc_samples, args.bc_epochs,
                            args.validation_episodes, args.test_episodes)):
        parser.error("training budgets must be positive")
    if args.timesteps < 1000 or len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        parser.error("use distinct nonnegative seeds and at least 1000 PPO steps")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning rate must be positive and finite")
    if args.test_start < 10000:
        parser.error("held-out test seeds must be separate from validation seeds")
    reports = [train_one(args, seed) for seed in args.seeds]
    write_json(args.output / "training_summary.json", {
        "seeds": args.seeds, "models": [{"seed": r["seed"], "steps": r["actual_ppo_steps"],
        "held_out_success_rate": r["held_out"]["success_rate"], "best_step": r["best_ppo_step"],
        "wall_seconds": r["total_wall_seconds"]} for r in reports]})


if __name__ == "__main__":
    main()
