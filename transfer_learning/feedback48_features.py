import torch
import torch.nn as nn
from torchvision import models

class Feedback48Features(nn.Module):
    def __init__(self, original_model):
        super(Feedback48Features, self).__init__()
        self.features = nn.Sequential(
            # stop at conv4
            *list(original_model.features.children())[:-2]
        )

    def forward(self, x):
        x = self.features(x)
        return x

