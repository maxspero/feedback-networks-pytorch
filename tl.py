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

feedback_net, optimizer, epoch = create_feedbacknet('feedback48', cuda)
epoch = load_checkpoint(feedback_net, optimizer, 'checkpoint20.pth.tar')


for p in feedback_net.parameters():
  p.requires_grad = False

#feedback_net.linear = nn.Linear(64, 10)
feedback_net.linear = nn.Sequential(
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
feedback_net.cuda()


optimizer = optim.Adam(feedback_net.linear.parameters())
criterion = nn.CrossEntropyLoss()

trainloader, valloader = load_train_data('cifar10')

for epoch in range(epoch_start, epochs):
  running_losses = np.zeros(feedback_net.num_iterations)
  running_loss = 0.0

  train_correct = np.zeros(feedback_net.num_iterations) 
  train_total = 0 
  for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    inputs, labels = Variable(inputs), Variable(labels)

    if cuda:
      inputs = inputs.cuda(device_id=0)
      labels = labels.cuda(device_id=0)

    optimizer.zero_grad()
    outputs = feedback_net(inputs)

    losses = [criterion(out, labels) for out in outputs]
    loss = sum(losses)
    loss.backward(retain_graph=True)
    optimizer.step()
    running_losses += [l.data[0] for l in losses]
    running_loss += loss.data[0]

    ## Print train accuracy
    train_total += labels.size(0)
    for it in range(feedback_net.num_iterations):
      _, predicted = torch.max(outputs[it].data, 1)
      train_correct[it] += (predicted == labels.data).sum()

    if i == 0:
      print('Epoch %d, iteration %d: loss=%f'% (epoch, i, running_loss))
      print('Running losses:')
      print([r for r in running_losses])
    elif i % 100 == 0:
      print('Epoch %d, iteration %d: loss=%f'% (epoch, i, running_loss/100.0))
      print('Running losses:')
      print([r/100.0 for r in running_losses])
      running_loss = 0.0
      running_losses = np.zeros(feedback_net.num_iterations)
  for it in range(feedback_net.num_iterations):
    train_acc = train_correct[it] / train_total
    print('Training accuracy for iteration %i: %f %%' % (it, 100 * train_acc))
    # Print val % accuracy
  correct = np.zeros(feedback_net.num_iterations) 
  total = 0 
  for data in valloader:
    inputs, labels = data
    inputs= Variable(inputs)

    if cuda:
      inputs = inputs.cuda(device_id=0)

      outputs = feedback_net(inputs)
      total += labels.size(0)
    for it in range(feedback_net.num_iterations):
      _, predicted = torch.max(outputs[it].data, 1)
      if cuda:
        predicted = predicted.cpu()
        correct[it] += (predicted == labels).sum()

  for it in range(feedback_net.num_iterations):
    val_acc = correct[it] / total
    print('Validation accuracy for iteration %i: %f %%' % (it, 100 * val_acc))

  save(feedback_net, optimizer, epoch)
print('done!')
