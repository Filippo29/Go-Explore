import gym
from time import sleep
from go_explore import Agent
import numpy as np

if __name__ == '__main__':
    file = np.fromfile("trajectory.txt", dtype=np.float64)
    print(file)
    #quit()
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    agent = Agent()
    agent.train()