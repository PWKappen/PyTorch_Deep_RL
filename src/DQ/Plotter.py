import matplotlib.pyplot as plt
import torch

plt.ion()

def plot_durations(episode_durations, episode_rewards, evaluation_rewards=None):
    plt.figure(1)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    rewards_t = torch.FloatTensor(episode_rewards)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration/Reward')
    plt.subplot(311)
    l1 = plt.plot(durations_t.numpy(), label='Duration')
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        l3 = plt.plot(means.numpy(), label='Mean Duration')
    plt.subplot(312)
    l2 = plt.plot(rewards_t.numpy(), label='Reward')
    if len(durations_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        l4 = plt.plot(means.numpy(), label='Mean Reward')

    if evaluation_rewards is not None:
        ev_rewards_t = torch.FloatTensor(evaluation_rewards)
        plt.subplot(313)
        l5 = plt.plot(ev_rewards_t.numpy(), label='Evaluation Reward')
        if len(ev_rewards_t) >= 100:
            means = ev_rewards_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            l6 = plt.plot(means.numpy(), label='Mean Evaluation Reward')

    plt.pause(0.001)