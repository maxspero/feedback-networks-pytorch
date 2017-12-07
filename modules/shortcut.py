import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math

class Shortcut(nn.Module):
    
    def __init__(self, stride):        
        super(Shortcut, self).__init__()

        
    def reset_state(self):
        self.last_cell = None
        self.last_h = None
    
    def forward(self, x):
        if self.last_cell is None:
            self.last_cell = Variable(torch.zeros(
                (x.size(0), self.output_size, int(x.size(2)/self.stride), 
                 int(x.size(3)/self.stride))
            ))
            if x.is_cuda:
                self.last_cell = self.last_cell.cuda()
        if self.last_h is None:
            self.last_h = Variable(torch.zeros(
                (x.size(0), self.output_size, int(x.size(2)/self.stride), 
                 int(x.size(3)/self.stride))
            ))
            if x.is_cuda:
                self.last_h = self.last_h.cuda()
        h = self.last_h
        c = self.last_cell
        
        # input gate
        input_x = self.batchnorm_i_x(self.conv_i_x(x))
        input_h = self.batchnorm_i_h(self.conv_i_h(h))
        input_gate = F.sigmoid(input_x + input_h)
        
        # forget gate
        forget_x = self.batchnorm_f_x(self.conv_f_x(x))
        forget_h = self.batchnorm_f_h(self.conv_f_h(h))
        forget_gate = F.sigmoid(forget_x + forget_h)
        
        # forget gate
        cell_x = self.batchnorm_c_x(self.conv_c_x(x))
        cell_h = self.batchnorm_c_h(self.conv_c_h(h))
        cell_intermediate = F.tanh(cell_x + cell_h) # g
        cell_gate = (forget_gate * c) + (input_gate * cell_intermediate)
        
        # output gate
        output_x = self.batchnorm_o_x(self.conv_o_x(x))
        output_h = self.batchnorm_o_h(self.conv_o_h(h))
        output_gate = F.sigmoid(output_x + output_h)
        
        next_h = output_gate * F.tanh(cell_gate)
        self.last_cell = cell_gate
        self.last_h = next_h
        
        return next_h
