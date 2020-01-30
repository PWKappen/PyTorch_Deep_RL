import torch.nn as nn
import torch.nn.functional as F

class BasicImageNetwork(nn.Module):
    def __init__(self, output_size):
        super(BasicImageNetwork, self).__init__()
        # self.conv1 = nn.Conv2d(2, 32, kernel_size=5, stride=2)
        # self.bn1 = nn.BatchNorm2d(32)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        # self.bn2 = nn.BatchNorm2d(64)
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=2)
        # self.bn3 = nn.BatchNorm2d(64)
        # self.fc1 = nn.Linear(1024*2, 256)
        # self.head = nn.Linear(256, output_size)
        # self.output_size = output_size

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        nn.init.xavier_uniform(self.conv1.weight)
        nn.init.constant(self.conv1.bias, 0)
        #self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        nn.init.xavier_uniform(self.conv2.weight)
        nn.init.constant(self.conv2.bias, 0)
        #self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        nn.init.xavier_uniform(self.conv3.weight)
        nn.init.constant(self.conv3.bias, 0)
        #self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(3136, 512)
        nn.init.xavier_uniform(self.fc1.weight)
        nn.init.constant(self.fc1.bias, 0)
        self.head = nn.Linear(512, output_size)
        nn.init.xavier_uniform(self.head.weight)
        nn.init.constant(self.head.bias, 0)
        self.output_size = output_size



    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.head(x)