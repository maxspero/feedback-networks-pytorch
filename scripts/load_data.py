import torch
import torchvision
import torchvision.transforms as transforms

def get_transform():
    return transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def load_train_data():
    transform = get_transform()
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, 
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
    return trainloader

def load_test_data():    
    transform = get_transform()
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, 
                                            transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)
    return testloader
