import gym

class Montezuma(gym.Env):
    def __init__(self, start_position=None, frame_skip=4, render_mode=None, deterministic=True, target_reward=99999999):
        env_name = 'MontezumaRevengeDeterministic-v4' if deterministic else 'MontezumaRevenge-v4'
        if render_mode is not None:
            self.env = gym.make(env_name, render_mode=render_mode)
        else:
            self.env = gym.make(env_name)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.frame_skip = frame_skip
        self.start_position = start_position # start position is a tuple of (state, checkpoint)
        self.target_reward = target_reward

        self.total_reward = 0
    
    def step(self, action):
        step_reward = 0
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            step_reward += reward
        self.total_reward += step_reward
        return obs, step_reward, terminated or truncated or info['lives'] < 6 or self.total_reward >= self.target_reward, info
    
    def restore_state(self, state):
        self.env.restore_state(state)
    
    def clone_state(self):
        return self.env.clone_state()
    
    def reset(self):
        if self.start_position is not None:
            self.env.reset()
            self.env.restore_state(self.start_position[1])
            return self.start_position[0]
        else:
            return self.env.reset()[0]
    
    def render(self):    
        self.env.render()