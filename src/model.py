from re import I
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, n_state, n_action):
        super().__init__()

        self.ll1 = nn.Linear(n_state, out_features=618)
        self.ll2 = nn.Linear(in_features=618, out_features=464)
        self.out_features = nn.Linear(in_features=464, out_features=n_action)

    def forward(self, input_t):
        input_t = F.relu(self.ll1(input_t))
        input_t = F.relu(self.ll2(input_t))
        input_t = self.out_features(input_t)
        return input_t
