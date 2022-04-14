from re import I
import torch.nn as nn
import torch.nn.functional as F
import torch


class DQN(nn.Module):
    def __init__(self, n_state, n_action):
        super().__init__()
        #self.emb = nn.Embedding(n_state, 10)

    #     super().__init__()
    #     self.conv1 = nn.Conv2d(3, 6, 5)
    #     self.pool = nn.MaxPool2d(2, 2)
    #     self.conv2 = nn.Conv2d(6, 16, 5)
    #     self.fc1 = nn.Linear(16 * 5 * 5, 120)
    #     self.fc2 = nn.Linear(120, 84)
    #     self.fc3 = nn.Linear(84, 10)

    # def forward(self, x):
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = torch.flatten(x, 1) # flatten all dimensions except batch
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return x


        self.ll1 = nn.Linear(n_state, out_features=243)
        self.ll2 = nn.Linear(in_features=243, out_features=393)
        self.ll3 = nn.Linear(in_features=393, out_features=635)
        self.ll4 = nn.Linear(in_features=635, out_features=1028)
        self.ll5 = nn.Linear(in_features=1028, out_features=1663)
        self.out_features = nn.Linear(in_features=1663, out_features=n_action)

    def forward(self, input_t):
        input_t = F.relu(self.ll1(input_t))
        input_t = F.relu(self.ll2(input_t))
        input_t = F.relu(self.ll3(input_t))
        input_t = F.relu(self.ll4(input_t))
        input_t = F.relu(self.ll5(input_t))
        input_t = self.out_features(input_t)
        return input_t
