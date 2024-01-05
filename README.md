# Go-Explore, exploration in Montezuma's Revenge
### Filippo Ansalone, Reinforcement Learning 2023

Reimplementation of https://arxiv.org/abs/1901.10995  
Environment: Montezuma's Revenge https://www.gymlibrary.dev/environments/atari/montezuma_revenge/  
Task: first build trajectories by randomly exploring a deterministic environment and then learn imitating the best trajectory.

## Installation
To install the needed packages:
```bash
pip install -r requirements.txt
```

## Phase 1
To run the phase 1 of the algorithm just execute:
```bash
python main.py --phase1
```
This process will produce the best trajectory found for each checkpoint (points where a new reward is collected) as files named best_trajectory_rew<reward>_dist<steps>.npy.

## Phase 2
To run the phase 2:
```bash
python main.py --phase2
```
Additionally, you can add the following arguments (optionally):
```bash
python main.py --phase1 --trajectory <string> --startpoint <int> --maxtimesteps <int> --patience <int>
```
trajectory: filename of the trajectory to imitate  
startpoint: starting point from where the robustification  
maxtimesteps: the maximum amount of timesteps for each robustification iteration  
patience: if set to -1, the behavior remains unchanged. If set to a positive scalar, the maximum number of iterations without any improvement at a specific point in the trajectory; if this limit is reached, then the algorithm is repeated from the previous point of the trajectory.  

The process will terminate once the policy is well performing from the initial state of the environment. As the paper states, the phase 2 is not guaranteed to converge both to a solution or an optimal solution for a given trajectory