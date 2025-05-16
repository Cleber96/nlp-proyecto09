
import torch.nn as nn
from .model_nat_base import NATBase

class FertilityPredictor(nn.Module):
    def __init__(self, hidden_size, max_fertility=10):
        super().__init__()
        self.linear = nn.Linear(hidden_size, max_fertility)

    def forward(self, encoder_hidden):
        return self.linear(encoder_hidden)