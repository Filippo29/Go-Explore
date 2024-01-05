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
    parser.add_argument('--trajectory', type=str, help='Input path to trajectory')
    parser.add_argument('--startpoint', type=int, help='Input start point for robustification phase')
    parser.add_argument('--maxtimesteps', type=int, help='Input max number of timesteps for each robustification step')
    parser.add_argument('--patience', type=int, help='Input max number of robustification phases without improvement before the starting point is changed')
    parser.add_argument('--policy', type=str, help='Input path to policy checkpoint')
    args = parser.parse_args()

    agent = Agent()
    if args.phase1:
        agent.explore()
    if args.phase2:
        traj_filename = "trajectory.npy" if not args.trajectory else args.trajectory
        start_point = None if not args.startpoint else args.startpoint
        max_timesteps = 1e2 if not args.maxtimesteps else args.maxtimesteps
        patience = 10 if not args.patience else args.patience
        transforms = None
        trajectory = build_trajectory(np.fromfile(traj_filename, dtype=np.float64).astype(int), transforms)
        robustification(trajectory, transforms, start_point=start_point, max_timesteps=max_timesteps, patience=patience)
    if args.test:
        from stable_baselines3 import PPO
        transforms = None
        env = Montezuma(deterministic=False, render_mode='human')
        policy_filename = "montezuma_save" if not args.policy else args.policy
        policy = PPO.load(policy_filename, env=env, device="mps")
        state = env.reset(0)[0]
        for _ in range(500):
            action = policy.predict(state)[0]
            next_state, reward, done, _, info = env.step(action)
            if done:
                break
            state = next_state 