import torch
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

def get_device():
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    if torch.cuda.is_available():
        device = "cuda"
    print("Using device:", device)
    return device

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning) # silence deprecation warning of np.bool8 in gym

    parser = argparse.ArgumentParser()
    parser.add_argument('-t1', '--test1', action='store_true')
    parser.add_argument('-t2', '--test2', action='store_true')
    parser.add_argument('-p1', '--phase1', action='store_true')
    parser.add_argument('-p2', '--phase2', action='store_true')
    parser.add_argument('-r', '--render', action='store_true')
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
        traj_filename = args.trajectory
        if traj_filename is None:
            print("Error: trajectory must be specified")
            quit()
        start_point = args.startpoint
        max_timesteps = 2e3 if not args.maxtimesteps else args.maxtimesteps
        patience = 10 if not args.patience else args.patience
        transforms = None
        trajectory = build_trajectory(np.fromfile(traj_filename, dtype=np.float64).astype(int), transforms)

        device = get_device()

        robustification(trajectory, transforms, start_point=start_point, max_timesteps=max_timesteps, patience=patience, device=device)
    if args.test1:
        traj_filename = args.trajectory
        if traj_filename is None:
            print("Error: trajectory must be specified")
            quit()
        actions = np.fromfile(traj_filename, dtype=np.float64)
        env = Montezuma(frame_skip=4, render_mode='human' if args.render else None)
        env.reset()
        tot_reward = 0
        for action in actions:
            _, reward, done, _, _ = env.step(int(action))
            tot_reward += reward
            if done:
                break
        print("Total reward: ", tot_reward)
    if args.test2:
        from stable_baselines3 import PPO
        transforms = None
        start_point = args.startpoint
        traj_filename = args.trajectory
        if (start_point is None and traj_filename is not None) or (start_point is not None and traj_filename is None):
            print("Error: both start point and trajectory must be specified")
            quit()
        if start_point is None:
            env = Montezuma(deterministic=False, render_mode='human' if args.render else None)
        else:
            trajectory = build_trajectory(np.fromfile(traj_filename, dtype=np.float64).astype(int), transforms)
            env = Montezuma(start_position=(trajectory[start_point][0], trajectory[start_point][2]), deterministic=False, render_mode='human' if args.render else None)
        policy_filename = "montezuma_save" if not args.policy else args.policy
        policy = PPO.load(policy_filename, env=env, device=get_device())
        state = env.reset(0)[0]
        tot_reward = 0
        max_timesteps = 500 if not args.maxtimesteps else args.maxtimesteps
        for _ in range(max_timesteps):
            action = policy.predict(state)[0]
            next_state, reward, done, _, _ = env.step(action)
            tot_reward += reward
            if done:
                break
            state = next_state 
        print("Total reward: ", tot_reward)