import torch 
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import numpy as np

from scripts.pascal import VOCSegmentationClasses


import torch.optim as optim

cuda = True
no_checkpoints = False
epoch_start = 0
epochs = 5

# remove fc -1
model49 = torchvision.models.resnet50(pretrained=True)
model49.fc = nn.ReLU()
model49.cuda()
# remove 3x3 and fc -10
model40 = torchvision.models.resnet50(pretrained=True)
model40.layer4 = nn.ReLU()
model40.avgpool = nn.AvgPool2d(14, stride=1)
model40.fc = nn.ReLU()
model40.cuda()
# remove 6x3 3x3 and fc -28
model22 = torchvision.models.resnet50(pretrained=True)
model22.layer3 = nn.ReLU()
model22.layer4 = nn.ReLU()
model22.avgpool = nn.AvgPool2d(28, stride=1)
model22.fc = nn.ReLU()
model22.cuda()
# remove 4x3 6x3 3x3 and fc -40
model10 = torchvision.models.resnet50(pretrained=True)
model10.layer2 = nn.ReLU()
model10.layer3 = nn.ReLU()
model10.layer4 = nn.ReLU()
model10.avgpool = nn.AvgPool2d(56, stride=1)
model10.fc = nn.ReLU()
model10.cuda()

print('models loaded')

models = [model10, model22, model40, model49]

def to_32_to_224():
    return transforms.Compose(
    [transforms.ToTensor(), transforms.Scale(32), transforms.Scale(224), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def load_train_data(batch_size=32, train_percent = 0.9):
    transform = to_32_to_224()
    trainset = VOCSegmentationClasses(root='./data', train=True, download=True, 
                                            transform=transform)

    val_split = int(len(trainset))
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(0, val_split))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=4, sampler=train_sampler)
    return trainloader

trainloader = load_train_data()

all_outputs = np.zeros((4, len(trainloader)*32, 256))
all_labels = np.zeros((len(trainloader)*32, 20))

print('dataset loaded!')
print(len(trainloader))

for i, data in enumerate(trainloader, 0):
  inputs, labels = data
  inputs = Variable(inputs, volatile=True)

  if cuda:
    inputs = inputs.cuda(device_id=0)

  for it in range(len(models)):
    o = models[it](inputs)
    npo = o.data.cpu().numpy()
    all_outputs[it,i*32:(i*32)+npo.shape[0]] = npo
  npl = labels.numpy()
  all_labels[i*32:(i*32)+npl.shape[0]] = npl
  print(i)

np.save('extracted_features', all_outputs)
np.save('labels', all_labels)
