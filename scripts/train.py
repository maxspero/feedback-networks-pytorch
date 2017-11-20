import torch 
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import numpy as np

from .create_model import create_feedbacknet
from .create_model import save
from .create_model import load_checkpoint
from .load_data import load_train_data
from .load_data import load_test_data

def train(checkpoint=None, cuda=False, epochs=20, dataset='cifar100', no_checkpoints=False):
    feedback_net, optimizer, epoch_start = create_feedbacknet('feedback48', cuda)

    if checkpoint is not None:
        epoch_start = load_checkpoint(feedback_net, optimizer, checkpoint)

    criterion = nn.CrossEntropyLoss()

    trainloader = load_train_data(dataset)
    
    for epoch in range(epoch_start, epochs):
        running_losses = np.zeros(feedback_net.num_iterations)
        running_loss = 0.0
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
        if not no_checkpoints:
            save(feedback_net, optimizer, epoch)
    print('done!')

def test(checkpoint=None, cuda=False, test_network=None, dataset='cifar100'):
    feedback_net, optimizer, epoch_start = create_feedbacknet('feedback48', cuda)
    if checkpoint is not None:
        epoch_start = load_checkpoint(feedback_net, optimizer, checkpoint)
    if test_network is not None:
        feedback_net = test_netowrk
    testloader = load_test_data(dataset)

    correct = np.zeros(feedback_net.num_iterations)
    total = 0
    for data in testloader:
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
            correct[i] += (predicted == labels).sum()

    for i in range(feedback_net.num_iterations):
      print('Accuracy for iteration %i: %f %%' % (i, 100 * correct[i] / total))
