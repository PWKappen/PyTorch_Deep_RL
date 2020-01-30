import Agents
import Models
from Memory import Memory
import Plotter

import gym
import torch
import numpy as np
from itertools import count

BATCH_SIZE = 64
NUM_UPDATES = 20
LEARNING_RATE_POLICY = 0.0003
LEARNING_RATE_VALUE = 0.0003
OPTIMIZER = torch.optim.Adam
GAMMA = 0.99
TAU = 0.95
CLIP_EPSILON = 0.2

NUM_INPUTS = 2
NUM_OUTPUTS = 1
GYM_ENVIRONMENT = 'MountainCarContinuous-v0'
NUM_EPISODES = 1000000
NUM_EPISODES_PER_TRAIN = 10

agent = Agents.PPO(Models.BasicDistributionNetwork, Models.ValueNetwork, NUM_INPUTS, NUM_OUTPUTS, OPTIMIZER,
                   LEARNING_RATE_POLICY, LEARNING_RATE_VALUE, BATCH_SIZE, NUM_UPDATES, GAMMA, TAU, CLIP_EPSILON)


env = gym.make(GYM_ENVIRONMENT)
print(env.action_space)
print(env.action_space.high)
print(env.action_space.low)

print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

episode_durations = []
episode_rewards = []

total_steps = 0
for i_episode in range(NUM_EPISODES):
    memory = Memory()

    num_steps = 0
    reward_batch = 0
    num_episodes = 0
    while num_steps < NUM_EPISODES_PER_TRAIN:
        total_steps += 1
        num_steps += 1
        state = env.reset()

        reward_sum = 0
        for t in count():
            if total_steps % 50 == 0:
                env.render()
            action = agent.get_action(state)
            action = action.cpu().data[0].numpy()
            next_state, reward, done, _ = env.step(action)
            reward_sum += reward

            mask = 1
            if done:
                mask = 0
            memory.push(state, np.array([action]), mask, reward)

            state = next_state
            if done:
                episode_durations.append(t+1)
                episode_rewards.append(reward_sum)
                Plotter.plot_durations(episode_durations, episode_rewards)
                break

    batch = memory.sample()
    agent.train_step(batch)