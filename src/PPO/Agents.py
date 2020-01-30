import torch
from torch.autograd import Variable
import numpy as np
from numpy import random

class PPO(object):
    def __init__(self, policy_network, value_network, input_size, output_size, optimizer, learning_rate_policy,
                 learning_rate_value, batch_size, num_updates, gamma, tau, clip_epsilon):
        self.batch_size = batch_size
        self.num_updates = num_updates

        self.input_size = input_size
        self.output_size = output_size

        self.gamma = gamma
        self.tau = tau
        self.clip_epsilon = clip_epsilon
        self.learning_rate_policy = learning_rate_policy
        self.learning_rate_value = learning_rate_value

        self.FloatTensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor
        self.ByteTensor = torch.cuda.ByteTensor
        self.Tensor = self.FloatTensor

        self.PI = self.Tensor([3.1415926])

        self.policy_network = policy_network(self.input_size, self.output_size).cuda()
        self.old_policy_network = policy_network(self.input_size, self.output_size).cuda()

        self.replace_network()

        self.value_network = value_network(self.input_size).cuda()

        self.policy_optimizer = optimizer(self.policy_network.parameters(), lr=learning_rate_policy)
        self.value_optimizer = optimizer(self.value_network.parameters(), lr=learning_rate_value)


    def replace_network(self):
        self.old_policy_network.load_state_dict(self.policy_network.state_dict())

    def get_action(self, state):
        state = Variable(torch.from_numpy(state).unsqueeze(0).type(self.Tensor))
        mean, _, std = self.policy_network(state)
        action = torch.normal(mean, std)
        return action

    def get_density(self, action, mean, log_std, std):
        var = std.pow(2)
        log_density = -(action - mean).pow(2) / (2 * var) - 0.5 * torch.log(2 * Variable(self.PI)) - log_std
        return log_density.sum(1)

    def train_step(self, batch):
        rewards = self.Tensor(batch.reward)
        masks = self.Tensor(batch.mask)
        con = np.concatenate(batch.action,0)
        actions = torch.from_numpy(con).type(self.Tensor)
        states = Variable(self.Tensor(batch.state))
        values = self.value_network(states)


        returns = self.Tensor(actions.size(0), 1)
        deltas = self.Tensor(actions.size(0), 1)
        advantages = self.Tensor(actions.size(0), 1)

        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(rewards.size(0))):
            returns[i] = rewards[i] + self.gamma * prev_return * masks[i]
            deltas[i] = rewards[i] + self.gamma * prev_value * masks[i] - values.data[i]
            advantages[i] = deltas[i] + self.gamma * self.tau * prev_advantage * masks[i]
            prev_return = returns[i, 0]
            prev_value = values.data[i, 0]
            prev_advantage = advantages[i, 0]

        targets = Variable(returns)

        num_total = len(batch.mask)

        action_var = Variable(actions)

        self.replace_network()

        action_means_old, action_log_stds_old, action_stds_old = self.old_policy_network(states)
        log_prob_old = self.get_density(action_var, action_means_old, action_log_stds_old, action_stds_old)

        # backup params after computing probs but before updating new params

        advantages = (advantages - advantages.mean()) / advantages.std()
        advantages_var = Variable(advantages)

        total_value_los = Variable(torch.zeros(1).type(self.FloatTensor))
        total_policy_los = Variable(torch.zeros(1).type(self.FloatTensor))
        for i in range(self.num_updates):
            elements = random.choice(num_total,self.batch_size).astype(int)
            ind = torch.from_numpy(elements).type(self.LongTensor)
            val = self.value_network(states[ind])
            self.value_optimizer.zero_grad()
            value_loss = (val-targets[ind]).pow(2.).mean()
            total_value_los += value_loss
            value_loss.backward()
            self.value_optimizer.step()

            action_means, action_log_stds, action_stds = self.policy_network(states[ind])
            log_prob_cur = self.get_density(action_var[ind], action_means, action_log_stds, action_stds)

            self.policy_optimizer.zero_grad()
            ratio = torch.exp(log_prob_cur - log_prob_old[ind].detach())
            tmp = advantages_var[ind]
            surr1 = torch.unsqueeze(ratio,-1) * tmp
            surr2 = torch.unsqueeze(torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon), -1) * advantages_var[ind]
            policy_surr = -torch.min(surr1, surr2).mean()
            total_policy_los += policy_surr
            policy_surr.backward()

            #torch.nn.utils.clip_grad_norm(self.policy_network.parameters(), 40)

            self.policy_optimizer.step()
        print(total_value_los)
        print(total_policy_los)
        print('\n')