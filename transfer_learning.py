import torch 
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import numpy as np

from scripts.create_model import create_feedbacknet
from scripts.create_model import save
from scripts.create_model import load_checkpoint
from scripts.load_data import load_train_data
from scripts.load_data import load_test_data
from transfer_learning.feedback48_features import Feedback48Features

cuda = False

feedback_net, optimizer, epoch = create_feedbacknet('feedback48', cuda)
epoch = load_checkpoint(feedback_net, optimizer, 'checkpoint20.pth.tar')


optimizer = optim.Adam(feedback_net.parameters())

feedback_net = Feedback48Features(feedback_net)
print(feedback_net)