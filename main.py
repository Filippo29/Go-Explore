import gym
from time import sleep
from go_explore import Agent
import numpy as np
import argparse

from environment import Montezuma

from policy import robustification

def build_trajectory(actions, transforms):
    env = Montezuma(frame_skip=4)
    state = env.reset(0)[0]
    trajectory = []
    for i in range(len(actions)):
        action = actions[i]
        next_state, reward, done, _, info = env.step(action)
        trajectory.append((state, action, env.clone_state(), reward))
        if done:
            break
        state = next_state
    return trajectory

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning) # silence deprecation warning of np.bool8 in gym

    parser = argparse.ArgumentParser(description='Run training and evaluation')
    parser.add_argument('-t', '--test', action='store_true')
    parser.add_argument('-p1', '--phase1', action='store_true')
    parser.add_argument('-p2', '--phase2', action='store_true')
    args = parser.parse_args()

    agent = Agent()
    if args.phase1:
        agent.train()
    if args.phase2:
        transforms = None
        trajectory = build_trajectory(np.fromfile("best_trajectory_rew100.0_dist201.txt", dtype=np.float64).astype(int), transforms)
        robustification(trajectory, transforms)
    if args.test:
        from stable_baselines3 import PPO
        transforms = None
        env = Montezuma(deterministic=False, render_mode='human')
        policy = PPO.load("montezuma_save", env=env, device="mps")
        state = env.reset(0)[0]
        for _ in range(500):
            env.render()
            action = policy.predict(state)[0]
            next_state, reward, done, _, info = env.step(action)
            if done:
                break
            state = next_state 