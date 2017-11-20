import torch
import torch.nn as nn
from torchvision import models


class AlexNetConv4(nn.Module):
    def __init__(self, original_model):
        super(AlexNetConv4, self).__init__()
        self.features = nn.Sequential(
            # stop at conv4
            *list(original_model.features.children())[:-3]
        )
    def forward(self, x):
        x = self.features(x)
        return x

def create_model():
    original_model = models.alexnet(pretrained=True)
    model = AlexNetConv4(original_model)
    return model