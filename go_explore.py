import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Grayscale, Compose
import numpy as np
import gym
from random import random
from time import sleep

from environment import Montezuma

class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.transforms = Compose([ToTensor(), Grayscale()])
        env = gym.make('MontezumaRevengeDeterministic-v4')
        self.action_space = env.action_space.n
        
    def forward(self):
        pass

    def process_state(self, state):
        state = self.transforms(state).unsqueeze(0)
        state = F.interpolate(state, size=(11, 8), mode='area')
        state = torch.round(state * 8).int()
        return state
    
    def random_action(self, last_action=-1):
        if last_action == -1:
            return np.random.randint(self.action_space)
        prob_per_action = 0.05 / (self.action_space-1)
        prob_distribution = np.zeros(self.action_space)
        prob_distribution.fill(prob_per_action)
        prob_distribution[last_action] = 0.95
        return np.random.choice(self.action_space, p=prob_distribution)
    
    def train(self, save_trajectories=True):
        # phase 1: explore
        self.cells = CellsManager()
        env = Montezuma()
        next_state = env.reset()

        import matplotlib.pyplot as plt
        self.cells.add(self.process_state(next_state), Cell(env.clone_state(), Trajectory(start_cell=-1))) # add initial state
        print("starting, cells: ", self.cells.size())

        best_cell_reward = 0
        best_cell_distance = 0
        
        while True:
            # explore
            env = Montezuma()
            env.reset()
            start_cell_index, processed_state, start_cell = self.cells.sample_cell()
            env.restore_state(start_cell.checkpoint)
            trajectory = Trajectory(start_cell=start_cell_index)
            finished = False
            action = self.random_action()
            while not finished:
                action = self.random_action(action)
                reward = 0
                next_frame, reward, finished, info = env.step(action)
                if finished:
                    break
                next_state = next_frame
                trajectory.add(processed_state, action, reward) # add last processed state with the action done in it and the obtained reward
                processed_state = self.process_state(next_state)

                index, cell = self.cells.contains(processed_state)
                if index == -1:
                    new_reward = start_cell.reward_from_start + trajectory.sum_rewards()
                    new_distance = start_cell.distance_from_start + trajectory.size()
                    new_cell = Cell(env.clone_state(), trajectory, new_distance, new_reward)
                    self.cells.add(processed_state, new_cell)
                    print("Added new cell with reward: ", new_reward, "after ", new_distance, "steps. New num cells: ", self.cells.size())
                    if save_trajectories:
                        if new_reward > best_cell_reward or (new_reward == best_cell_reward and new_distance < best_cell_distance):
                            print("Saved trajectory with reward: ", new_cell.reward_from_start, "after ", new_cell.distance_from_start, "steps.")
                            save_array(self.cells.get_trajectory_to_cell(new_cell), "best_trajectory_rew" + str(new_reward) + "_dist" + str(new_distance) + ".txt")
                            best_cell_reward = new_reward
                            best_cell_distance = new_distance
                    break # if found new cell, stop exploring
                else:
                    self.cells.times_visited[index] += 1

def save_array(array, filename="trajectory.txt"):
    np.array(array).tofile(filename)

class Trajectory:
    def __init__(self, start_cell):
        self.states = []
        self.actions = np.array([])
        self.rewards = np.array([])
        self.start_cell = start_cell # index of the cell where the trajectory starts

    def add(self, state, action, reward):
        self.states.append(state)
        self.actions = np.append(self.actions, action)
        self.rewards = np.append(self.rewards, reward)
    
    def sum_rewards(self):
        return np.sum(self.rewards)

    def size(self):
        return len(self.states)

class Cell:
    def __init__(self, checkpoint, trajectory, distance_from_start=0, reward_from_start=0):
        self.checkpoint = checkpoint
        self.trajectory = trajectory
        self.distance_from_start = distance_from_start
        self.reward_from_start = reward_from_start
    
    def update_trajectory(self, trajectory):
        self.trajectory = trajectory

class CellsManager:
    def __init__(self):
        self.states = [] # processed state of the cell
        self.cells = np.array([]) # cell object

        self.times_chosen = np.array([])
        self.times_visited = np.array([])

        self.eps1 = 0.001
        self.eps2 = 0.00001
        self.w = [0.1, 0.3] # wheights for times chosen and seen
        self.p = [1, 1]
    
    def add(self, state, cell):
        self.states.append(state)
        self.cells = np.append(self.cells, cell)
        self.times_chosen = np.append(self.times_chosen, 0)
        self.times_visited = np.append(self.times_visited, 0)
    
    def contains(self, state):
        # return index of state
        for i in range(len(self.states)):
            if torch.equal(self.states[i], state):
                return i, self.cells[i]
        return -1, None
    
    def get_cell(self, state):
        index = -1
        for i in range(len(self.states)):
            if torch.equal(self.states[i], state):
                index = i
        if index == -1:
            return None
        return index, self.cells[index]

    def size(self):
        return len(self.states)
    
    def sample_cell(self):

        index = np.random.choice(len(self.cells))
        self.times_chosen[index] += 1
        return index, self.states[index], self.cells[index]
    
    def get_cells_indexes_from_start_to_cell(self, cell):
        cells_indexes = []
        while True:
            if cell.trajectory.start_cell == -1:
                break
            else:
                cells_indexes.append(cell.trajectory.start_cell)
                cell = self.cells[cell.trajectory.start_cell]
        cells_indexes.reverse()
        return cells_indexes
    
    def get_trajectory_to_cell(self, cell):
        trajectories = []
        trajectory = cell.trajectory
        while True:
            trajectories.append(trajectory)
            if trajectory.start_cell == -1:
                break
            else:
                trajectory = self.cells[trajectory.start_cell].trajectory
        trajectories.reverse()
        tot_traj = np.array([])
        for traj in trajectories:
            tot_traj = np.concatenate((tot_traj, traj.actions), axis=0)
        return tot_traj