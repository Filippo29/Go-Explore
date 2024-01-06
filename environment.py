import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np

class Montezuma(gym.Env):
    def __init__(self, transforms=None, convert_grayscale=True, start_position=None, frame_skip=1, stack=4, render_mode=None, deterministic=True, max_steps=np.inf, target_reward=np.inf):
        env_name = 'MontezumaRevengeDeterministic-v4' if deterministic else 'MontezumaRevenge-v4'
        if render_mode is not None:
            self.env = gym.make(env_name, render_mode=render_mode)
        else:
            self.env = gym.make(env_name)
        self.stack = stack

        self.observation_space = Box(0, 255, (self.stack, 210, 160), dtype=np.uint8)
        self.action_space = self.env.action_space

        self.frame_skip = frame_skip
        self.start_position = start_position # start position is a tuple of (state, checkpoint)
        self.target_reward = target_reward

        self.transforms = transforms
        self.convert_grayscale = convert_grayscale

        self.last_observations = []

        self.max_steps = max_steps
        self.steps = 0

        self.total_reward = 0
    
    def step(self, action):
        step_reward = 0
        finished = False
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if info['lives'] < 6:
                #step_reward = -20
                finished = True
            step_reward += reward
            if terminated or truncated or self.total_reward >= self.target_reward:
                finished = True
            if len(self.last_observations) < self.stack:
                while len(self.last_observations) < self.stack:
                    self.last_observations.append(self.grayscale(obs))
            self.last_observations = self.last_observations[1:]
            obs = self.grayscale(obs)
            if self.transforms is not None:
                obs = self.transforms(obs)
            self.last_observations.append(obs)
        self.steps += 1
        self.total_reward += step_reward
        finished = finished or self.steps >= self.max_steps
        return np.stack(self.last_observations, dtype=np.float32), step_reward, finished, finished, info
    
    def restore_state(self, state):
        self.env.restore_state(state)
    
    def clone_state(self):
        return self.env.clone_state()
    
    def reset(self, seed=0):
        self.total_reward = 0
        self.steps = 0
        if self.start_position is not None:
            self.env.reset(seed=seed)
            self.env.restore_state(self.start_position[1])
            return self.start_position[0], {}
        else:
            obs = self.grayscale(self.env.reset(seed=seed)[0])
            if self.transforms is not None:
                obs = self.transforms(obs)
            observations = [obs for _ in range(self.stack)]
            return np.stack(observations, dtype=np.float32), {}
    
    def get_action_meanings(self):
        return self.env.get_action_meanings()
    
    def render(self):    
        self.env.render()

    def grayscale(self, rgb):
        if not self.convert_grayscale:
            return rgb
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray