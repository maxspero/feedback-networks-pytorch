import torch
import torchvision
import torchvision.transforms as transforms
from .pascal import VOCSegmentationClasses

def get_transform():
    return transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def load_train_data(dataset, batch_size=32, train_percent = 0.9):
    transform = get_transform()
    dataset = dataset.lower()
    if dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, 
                                                transform=transform)
    elif dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, 
                                                transform=transform)
    elif dataset == 'pascal': # pascal VOC 2012 
        trainset = VOCSegmentationClasses(root='./data', train=True, download=True, 
                                                transform=transform)
    elif dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, 
                                                transform=transform)
    else:
        raise ValueError('Invalid dataset')
    val_split = int(len(trainset) * train_percent)
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(0, val_split))
    validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(val_split, len(trainset)))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=4, sampler=train_sampler)
    valloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=4, sampler=validation_sampler)
    return trainloader, valloader

def load_test_data(dataset, batch_size=32):
    transform = get_transform()
    dataset = dataset.lower()
    if dataset == 'cifar100':
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, 
                                                transform=transform)
    elif dataset == 'cifar10':
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, 
                                                transform=transform)
    elif dataset == 'pascal': # pascal VOC 2012 
        testset = VOCSegmentationClasses(root='./data', train=False, download=True, 
                                                transform=transform)
    elif dataset == 'mnist':
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, 
                                                transform=transform)
    else:
        raise ValueError('Invalid dataset')
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    return testloader
