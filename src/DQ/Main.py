import gym
import DRLAlgorithms
import Models
from Plotter import *
from EnvFunctions import *
from itertools import count
import ReplayMemory
import Explorer
import torch.optim as optim

NUM_OUTPUS = 2
REPLAY_MEMORY_SIZE = 50000
BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 0.00025
EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 100000
NUM_ELEMENTS = 4
NUM_STEPS_WAIT = 20000

env = gym.make('Pong-v0')

DQ = DRLAlgorithms.DQ(Models.BasicImageNetwork, NUM_OUTPUS, optim.Adam, LEARNING_RATE, ReplayMemory.BasicMemory(REPLAY_MEMORY_SIZE),
                 Explorer.ExponentialExplorer(EPS_END, EPS_START, EPS_DECAY, NUM_STEPS_WAIT), BATCH_SIZE, GAMMA, True, 1000)

episode_durations = []
episode_rewards = []

num_episodes = 100000
step = 0
for i_episode in range(num_episodes):
    env.reset()
    img, reward, done, _= env.step(0)
    preivous_screen = [adapt_image(img),adapt_image(img),adapt_image(img),adapt_image(img)]
    new_screen = [adapt_image(img),adapt_image(img),adapt_image(img),adapt_image(img)]
    current_screen=img
    state = torch.cat(new_screen,dim=1)

    total_reward = 0

    for t in count():
        env.render()
        action = DQ.get_action(state)
        tot_reward = 0
        for i in range(NUM_ELEMENTS):
            #check if done
            last_screen, reward, done, _ = env.step(action[0, 0]+2)
            #new_screen[i] = adapt_image(np.maximum(last_screen, current_screen))
            new_screen[i] = adapt_image(last_screen)
            current_screen = last_screen
            tot_reward += reward
        total_reward += tot_reward
        reward = Tensor([tot_reward])

        if not done:
            next_state = torch.cat(new_screen,dim=1)
        else:
            next_state = None

        DQ.add_memory(state, action, next_state, reward)
        step += 1
        state = next_state

        if step > NUM_STEPS_WAIT:
            DQ.optimize_model()
        if done:
            episode_durations.append(t + 1)
            episode_rewards.append(total_reward)
            plot_durations(episode_durations, episode_rewards)
            break
