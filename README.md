# PyTorch_Deep_RL
Simple implementation of Deep Q Learning and Proximal Policy Optimization in PyTorch

##  Explanation
This repository consists of two projects. One is an implementation of Deep Q Leraning and the other of Proximal Policy Optimization.
The project was intendet for educational purposes and is used to train Deep Reinforcement Leraning agents on OpenAI environments.

##  Dependencies
To run the programs the following Python packages must be installed:
* PyTorch (preferable GPU)
* Numpy
* Matplotlib
* OpenAI Gym (including the atari environments when training on the Pong environment)
* PIL (when training on the Pong environement).

##  Usage
Depending on which algorithm should be used, the directory src/DQ for Deep Q Learning, or src/PPO for Proximal Policy Optimization, must be downloaded. Both projects contain a main.py file which must be executed to start training. The Deep Q Learning algorithm 
is trained on the Atari Pong environment and the Proximal Policy Optimization is trained on the Continuous Mountain Car environment.  
To change the used environment, change the gym environment string. Both projects contain NUM_OUTPUTS, which must be changed to the number of actions the environment expects as input. The type, continuous or discrete, must stay the same.   
PPO also contains a NUM_INPUTS variable, which must be changed to the state size of the environment. This is not necessary for Deep Q Learning, since it is supposed to always run on atari environments, that all have the same state size.
