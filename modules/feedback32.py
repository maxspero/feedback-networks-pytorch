import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import math
from .feedbackmodule import FeedbackConvLSTM

class FeedbackNet32(nn.Module):  # 4 physical depth, 8 iterations
    def __init__(self):
        super(FeedbackNet32, self).__init__()
        self.conv = nn.Conv2d(3, 16, 3, 1, 1)
        self.batchnorm = nn.BatchNorm2d(16)
        self.feedback_conv_lstm = FeedbackConvLSTM(
            16, [32, 32, 64, 64], [2, 1, 2, 1], 8, 3, 3
        )
        self.avg_pool = nn.AvgPool2d(8)
        self.linear = nn.Linear(64, 100)
        
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x_all = self.feedback_conv_lstm(x)
        x_finished = []
        for x_i in x_all:
            x_i = F.relu(x_i)
            x_i = self.avg_pool(x_i)
            x_i = x_i.view(-1, 64)
            x_i = self.linear(x_i)
            x_finished.append(x_i)
        return x_finished
