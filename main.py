import gym
from time import sleep
from go_explore import Agent
import numpy as np
import argparse
from torchvision.transforms import ToTensor, Grayscale, Compose
import torch

from environment import Montezuma

from policy import robustification, Policy

def build_trajectory(actions, transforms):
    env = Montezuma()
    state = env.reset()[0]
    trajectory = []
    for i in range(len(actions)):
        action = actions[i]
        next_state, reward, done, info = env.step(action)
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
        transforms = Compose([ToTensor(), Grayscale()])
        trajectory = build_trajectory(np.fromfile("best_trajectory_rew100.0_dist201.txt", dtype=np.float64).astype(int), transforms)
        policy = robustification(trajectory, transforms)
        torch.save(policy.state_dict(), "policy.pt")
    if args.test:
        transforms = Compose([ToTensor(), Grayscale()])
        env = Montezuma(render_mode='human')
        policy = Policy(env.action_space.n)
        policy.load_state_dict(torch.load("policy.pt"))
        state = env.reset()[0]
        for _ in range(500):
            env.render()
            action = policy(transforms(state).unsqueeze(0)).argmax().item()
            next_state, reward, done, info = env.step(action)
            if done:
                break
            state = next_state
            sleep(0.05)   