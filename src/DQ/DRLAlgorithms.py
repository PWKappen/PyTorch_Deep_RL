import torch
from torch.autograd import Variable
import torch.nn.functional as F
import ReplayMemory
import random
import numpy as np

class DQ(object):
    def __init__(self, model, output_size, optimizer, learning_rate, replay_memory, explorer, batch_size, gamma, use_target=False, target_update=None):
        self.batch_size = batch_size
        self.use_target = use_target
        self.gamma = gamma
        self.output_size = output_size

        self.memory = replay_memory
        self.explorer = explorer

        self.use_cuda = torch.cuda.is_available()
        #torch.backends.cudnn.benchmark = True
        self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if self.use_cuda else torch.LongTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_cuda else torch.ByteTensor
        self.Tensor = self.FloatTensor

        self.q_network = model(output_size)

        if self.use_cuda:
            self.q_network = self.q_network.cuda()

        if use_target:
            self.q_target_network = model(output_size).cuda()

            if self.use_cuda:
                self.q_target_network.cuda()
            self.q_target_network.load_state_dict(self.q_network.state_dict())
            #self.update_target()

        if target_update is not None:
            self.target_update = target_update

        self.loss = torch.nn.MSELoss()
        self.optimizer = optimizer(self.q_network.parameters(), lr=learning_rate, eps=1e-6)

        self.update_step = 0
        self.step = 0

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)

        batch = ReplayMemory.Transition(*zip(*transitions))

        non_final_maks = self.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]).type(torch.cuda.FloatTensor)/255.0)

        state_batch = Variable(torch.cat(batch.state).type(torch.cuda.FloatTensor)/255.0)
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))

        next_state_values = Variable(torch.zeros(self.batch_size).type(self.Tensor))

        if (self.use_target):
            # tmp = self.q_network(non_final_next_states).detach()
            # _, max_a = tmp.data.max(1, keepdim=True)
            # values = self.q_target_network(non_final_next_states).detach()
            # next_state_values[non_final_maks] = values.data.gather(1,max_a)
            next_state_values[non_final_maks] = self.q_target_network(non_final_next_states).detach().max(1)[0]
        else:
            next_state_values[non_final_maks] = self.q_network(non_final_next_states).max(1)[0].detach()



        state_action_values = self.q_network(state_batch).gather(1, action_batch)

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = self.loss(state_action_values, torch.unsqueeze(expected_state_action_values,-1))

        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.q_network.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        if (self.update_step % 500 == 0):
            print("loss: " + str(loss.cpu().data[0]))

        if self.use_target:
            if self.update_step % self.target_update == 0:
                self.update_target()

        self.update_step += 1

    def update_target(self):
        # for q,t in zip(self.q_network.parameters(), self.q_target_network.parameters()):
        #     t.data = t.data *( 1- self.target_update) + q.data * self.target_update
        self.q_target_network.load_state_dict(self.q_network.state_dict())

    def get_action(self, state, evaluation=False):
        sample = random.random()
        eps_threshold = self.explorer(self.step)
        self.step += 1
        if sample > eps_threshold or (evaluation and sample > 0.05):
            tmp = self.q_network(Variable(state, volatile=True).type(self.FloatTensor)/255)
            if self.step % 100 == 0:
                print(tmp.cpu().data[0])
            return tmp.data.max(1)[1].view(1, 1)
        else:
            return self.LongTensor([[random.randrange(self.output_size)]])


    def add_memory(self, *args):
        self.memory.push(*args)
