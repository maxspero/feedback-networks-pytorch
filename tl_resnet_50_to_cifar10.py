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
from multiprocessing import freeze_support

if __name__ == '__main__':
  freeze_support()



model50 = torchvision.models.resnet50(pretrained=True)
epoch = load_checkpoint(feedback_net, optimizer, 'checkpoint41_feedback2_cifar10.pth.tar')


for p in feedback_net.parameters():
  p.requires_grad = False

#feedback_net.linear = nn.Linear(64, 10)
feedback_net.linear = nn.Linear(256, 256)
feedback_net.output = nn.Sequential(
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 20),
)
feedback_net.cuda()

optimizer = optim.Adam([{'params': feedback_net.linear.parameters()}, {'params': feedback_net.output.parameters()}])

criterion = nn.MultiLabelSoftMarginLoss()
dataset = 'pascal'
trainloader, valloader = load_train_data(dataset)
testloader = load_test_data(dataset)
print('hi')
for epoch in range(epoch_start, epochs):
  running_losses = np.zeros(feedback_net.num_iterations)
  running_loss = 0.0

  train_correct = np.zeros(feedback_net.num_iterations) 
  train_total = 0 
  for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    inputs, labels = Variable(inputs), Variable(labels).float()

    if cuda:
      inputs = inputs.cuda(device_id=0)
      labels = labels.cuda(device_id=0)

    optimizer.zero_grad()
    outputs = feedback_net(inputs)

    losses = [criterion(out, labels) for out in outputs]
    loss = Variable(torch.from_numpy(np.zeros(1))).float().cuda()
    for it in range(len(losses)):
      loss += (gamma ** it) * losses[it]
    #loss = sum(losses)
    loss.backward(retain_graph=True)
    optimizer.step()
    running_losses += [l.data[0] for l in losses]
    running_loss += loss.data[0]

    train_total += labels.size(0)
    for i in range(feedback_net.num_iterations):
      _, predicted = torch.max(outputs[i].data, 1)
      if cuda:
          predicted = predicted.cpu()
      if dataset != 'pascal':
        train_correct[i] += (predicted == labels).sum()
      else:
        for p in range(predicted.size(0)):
          if (labels.data[p, predicted[p]] > 0):
            train_correct[i] += 1

    if i == 0:
      print('Epoch %d, iteration %d: loss=%f'% (epoch, i, running_loss))
      print('Running losses:')
      print([r for r in running_losses])
    elif i % 40 == 0:
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
    for i in range(feedback_net.num_iterations):
      _, predicted = torch.max(outputs[i].data, 1)
      if cuda:
          predicted = predicted.cpu()
      if dataset != 'pascal':
        correct[i] += (predicted == labels).sum()
      else:
        for p in range(predicted.size(0)):
          if labels[p, predicted[p]] > 0:
            correct[i] += 1

  for it in range(feedback_net.num_iterations):
    val_acc = correct[it] / total
    print('Validation accuracy for iteration %i: %f %%' % (it, 100 * val_acc))

  save(feedback_net, optimizer, epoch)
print('done!')


correct = np.zeros(feedback_net.num_iterations)
total = 0

feedback_net.train(True)
for data in testloader:
  inputs, labels = data
  inputs= Variable(inputs, volatile=True)
  if cuda:
      inputs = inputs.cuda(device_id=0)

  outputs = feedback_net(inputs)
  total += labels.size(0)
  for i in range(feedback_net.num_iterations):
      _, predicted = torch.max(outputs[i].data, 1)
      if cuda:
          predicted = predicted.cpu()
      if dataset != 'pascal':
        correct[i] += (predicted == labels).sum()
      else:
        for p in range(predicted.size(0)):
          if labels[p, predicted[p]] > 0:
            correct[i] += 1

for i in range(feedback_net.num_iterations):
  print('Accuracy for iteration %i: %f %%' % (i, 100 * correct[i] / total))
