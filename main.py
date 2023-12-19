import gym
from time import sleep
from go_explore import Agent
import numpy as np
import argparse
from torchvision.transforms import ToTensor, Grayscale, Compose
import torch

from policy import phase2, Policy

def build_trajectory(actions, transforms):
    env = gym.make('MontezumaRevengeDeterministic-v4')
    state = env.reset()[0]
    trajectory = []
    for i in range(len(actions)):
        action = actions[i]
        reward = 0
        for _ in range(4):
            next_state, frame_reward, done, _, info = env.step(action)
            reward += frame_reward
        trajectory.append((transforms(state), action, env.clone_state(), reward))
        if done:
            break
    return trajectory

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

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
        trajectory = build_trajectory(np.fromfile("best_trajectory_rew100.0_dist160.txt", dtype=np.float64).astype(int), transforms)
        policy = phase2(trajectory, transforms)
        torch.save(policy.state_dict(), "policy.pt")
    if args.test:
        transforms = Compose([ToTensor(), Grayscale()])
        env = gym.make('MontezumaRevenge-v4', render_mode='human')
        policy = Policy(env.action_space.n)
        policy.load_state_dict(torch.load("policy.pt"))
        state = env.reset()[0]
        for _ in range(500):
            env.render()
            action = policy(transforms(state).unsqueeze(0)).argmax().item()
            for i in range(4):
                next_state, reward, term, trunc, info = env.step(action)
                if term or trunc:
                    break
            state = next_state
            sleep(0.05)



#conda activate go_explore
#cd Desktop\rl_project\Go-Explore
#python main.py
