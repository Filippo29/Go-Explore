import gym
from time import sleep
from explore import Agent
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

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', action='store_true')
    parser.add_argument('-p1', '--phase1', action='store_true')
    parser.add_argument('-p2', '--phase2', action='store_true')
    parser.add_argument('--trajectorty', type=str, help='Input path to trajectory')
    parser.add_argument('--startpoint', type=int, help='Input start point for robustification phase')
    parser.add_argument('--policy', type=str, help='Input path to policy checkpoint')
    args = parser.parse_args()

    agent = Agent()
    if args.phase1:
        agent.explore()
    if args.phase2:
        if not args.trajectory:
            traj_filename = "trajectory.txt"
        else:
            traj_filename = args.trajectory
        start_point = None
        if args.startpoint:
            start_point = args.startpoint
        transforms = None
        trajectory = build_trajectory(np.fromfile(traj_filename, dtype=np.float64).astype(int), transforms)
        robustification(trajectory, transforms, start_point=start_point)
    if args.test:
        from stable_baselines3 import PPO
        transforms = None
        env = Montezuma(deterministic=False, render_mode='human')
        if not args.policy:
            policy_filename = "montezuma_save"
        else:
            policy_filename = args.policy
        policy = PPO.load(policy_filename, env=env, device="mps")
        state = env.reset(0)[0]
        for _ in range(500):
            action = policy.predict(state)[0]
            next_state, reward, done, _, info = env.step(action)
            if done:
                break
            state = next_state 