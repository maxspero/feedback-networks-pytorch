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

import torch.optim as optim

cuda = True
no_checkpoints = False
epoch_start = 0
epochs = 5

feedback_net, optimizer, epoch = create_feedbacknet('feedback48_4', cuda)
epoch = load_checkpoint(feedback_net, optimizer, 'checkpoint38_feedback4_cifar100.pth.tar')
feedback_net.dropout1 = nn.ReLU()
feedback_net.linear = nn.ReLU()
feedback_net.dropout2 = nn.ReLU()
feedback_net.output = nn.ReLU()
feedback_net.cuda()

trainloader, valloader = load_train_data('cifar10')

all_outputs = np.zeros((feedback_net.num_iterations, len(trainloader)*32, 256))
all_labels = np.zeros(len(trainloader)*32)

for i, data in enumerate(trainloader, 0):
  inputs, labels = data
  inputs = Variable(inputs, volatile=True)

  if cuda:
    inputs = inputs.cuda(device_id=0)

  outputs = feedback_net(inputs)
  for it, o in enumerate(outputs):
    npo = o.data.cpu().numpy()
    all_outputs[it,i*32:(i*32)+npo.shape[0]] = npo
  npl = labels.numpy()
  all_labels[i*32:(i*32)+npl.shape[0]] = npl

np.save('extracted_features', all_outputs)
np.save('labels', all_labels)
