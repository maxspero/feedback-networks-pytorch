import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import math
from modules.feedback32 import FeedbackNet32

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
  torch.save(state, filename)
  
if __name__ == '__main__':

  transform = transforms.Compose(
  [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, 
                                          transform=transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
  testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, 
                                          transform=transform)
  testloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=False, num_workers=2)

  feedback_net = FeedbackNet32()

  # use GPU
  feedback_net.cuda()

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(feedback_net.parameters())

  for epoch in range(3):
      running_losses = np.zeros(8)
      running_loss = 0.0
      for i, data in enumerate(trainloader, 0):
          inputs, labels = data
          inputs, labels = Variable(inputs), Variable(labels)
          
          optimizer.zero_grad()
          outputs = feedback_net(inputs)
          
          losses = [criterion(out, labels) for out in outputs]
          loss = sum(losses)
          
          loss.backward(retain_graph=True)
          optimizer.step()
          running_losses += [l.data[0] for l in losses]
          running_loss += loss.data[0]
          if i % 100 == 0:
              print('Epoch %d, iteration %d: loss=%f' % (epoch, i, running_loss/100))
              print('Running losses:')
              print([r/100.0 for r in running_losses])
              running_loss = 0.0
              running_losses = np.zeros(8)
      save_checkpoint({
          'epoch': epoch + 1,
          'state_dict': feedback_net.state_dict(),
          'optimizer' : optimizer.state_dict(),
      })

  print('done!')
