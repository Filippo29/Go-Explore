import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

from environment import Montezuma

def get_target_reward(trajectory, start_index):
    return sum([trajectory[i][3] for i in range(start_index, len(trajectory))])

def robustification(trajectory, transforms, start_point=None, max_timesteps=1e2, patience=10, device="cpu"):
    if start_point is None:
        start_point = len(trajectory)-10
    deterministic = False
    i = 0
    while start_point >= 0:
        print("Starting part ", i)
        print("Start point: ", start_point)
        target_reward = get_target_reward(trajectory, start_point)
        max_steps = 5*(len(trajectory)-start_point)
        print("Target reward: ", target_reward)
        
        start_position = (trajectory[start_point][0], trajectory[start_point][2]) if start_point > 0 else None
        args = {"start_position": start_position, 
                "target_reward": target_reward, 
                "frame_skip": 1,
                "max_steps": max_steps,
                "deterministic": deterministic}

        env = Montezuma(**args)

        # wrap the environment
        env = DummyVecEnv([lambda: env])
        if i == 0:
            if start_point < len(trajectory)-10:
                # start from a checkpoint
                model = PPO.load("montezuma_save", env=env, device=device)
            else:
                model = PPO("CnnPolicy", env, device=device, verbose=1)
        else:
            model.set_env(env)
        score = test(Montezuma, transforms, model, trajectory, start_point, target_reward, num_simulations=5, deterministic=deterministic, log=True)
        epochs_with_no_reward = 0
        while score < target_reward:
            model.learn(total_timesteps=int(max_timesteps), progress_bar=True)
            score = test(Montezuma, transforms, model, trajectory, start_point, target_reward, num_simulations=5, deterministic=deterministic, log=True)
            if score == 0:
                epochs_with_no_reward += 1
            else:
                epochs_with_no_reward = 0
            if patience != -1 and epochs_with_no_reward >= patience:
                break
        if patience != -1 and epochs_with_no_reward >= patience:
            model = PPO.load("montezuma_save", env=env, device=device)
            start_point += 1
            continue
        model.save("montezuma_save")
        
        test(Montezuma, transforms, model, trajectory, start_point, target_reward, num_simulations=1, deterministic=deterministic, log=True, render_mode="human")

        if start_point > 0 and start_point < 5:
            start_point = 0
        else:
            start_point -= 5

        i += 1

def test(env_class, transforms, model, trajectory, start_point, target_reward, max_steps=np.inf, num_simulations=5, deterministic=True, render_mode=None, log=True):
    env = env_class(render_mode=render_mode, transforms=transforms, max_steps=max_steps, target_reward=target_reward, start_position=(trajectory[start_point][0], trajectory[start_point][2]), deterministic=deterministic)
    predictions = [0 for _ in range(env.action_space.n)]

    max_steps = 5*(len(trajectory)-start_point)
    done = False
    tot_reward = 0
    for _ in range(num_simulations):
        obs = env.reset(0)[0]
        for _ in range(max_steps):
            action = model.predict(obs)[0]
            predictions[action] += 1
            obs, reward, done, _, _ = env.step(action)
            tot_reward += reward
            if done:
                break
    if log:
        print("Average reward: ", tot_reward / num_simulations)
        print("Predictions: ", predictions)
    return tot_reward / num_simulations