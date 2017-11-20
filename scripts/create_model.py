import torch 
import torch.nn as nn
import torch.optim as optim
from modules.feedback32 import FeedbackNet32
from modules.feedback48 import FeedbackNet48


def save(model, optimizer, epoch):
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
    }, ('checkpoint%d.pth.tar' % epoch))
    print('Checkpoint %d saved successfully!' % epoch) 


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
  

def load_checkpoint(model, optimizer, filename='checkpoint.pth.tar'):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']


def create_feedbacknet(model, cuda):
    if model == 'feedback32':
        feedback_net = FeedbackNet32()
    elif model == 'feedback48':
        feedback_net = FeedbackNet48()


    # use GPU
    if cuda:
        feedback_net.cuda(device_id=0)

    optimizer = optim.Adam(feedback_net.parameters())

    return feedback_net, optimizer, 0
