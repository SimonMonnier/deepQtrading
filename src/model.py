from re import I
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, n_state, n_action):
        super().__init__()
        #self.emb = nn.Embedding(n_state, 10)
        self.ll1 = nn.Linear(n_state, out_features=168)
        self.ll2 = nn.Linear(in_features=168, out_features=272)
        self.ll3 = nn.Linear(in_features=272, out_features=441)
        self.ll4 = nn.Linear(in_features=441, out_features=712)
        self.ll5 = nn.Linear(in_features=712, out_features=1153)
        self.ll6 = nn.Linear(in_features=1153, out_features=1866)
        self.out_features = nn.Linear(in_features=1866, out_features=n_action)

    def forward(self, input_t):
        input_t = F.relu(self.ll1(input_t))
        input_t = F.relu(self.ll2(input_t))
        input_t = F.relu(self.ll3(input_t))
        input_t = F.relu(self.ll4(input_t))
        input_t = F.relu(self.ll5(input_t))
        input_t = F.relu(self.ll6(input_t))
        input_t = self.out_features(input_t)
        return input_t
