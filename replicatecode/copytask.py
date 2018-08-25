import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from data_utils import *
from torch.nn import Dropout
from syntheticmodel import *
from logger import Logger
from torch import nn
from data_utils import *
from logger import Logger
num_batches  = 600000
num_layers = 3
batch_size = 1
num_units = 8
seq_width = 8
min_len = 2
max_len = 20
ftype = torch.FloatTensor
dtype = torch.cuda.FloatTensor
logger = Logger('./copy_logs')
init = (Variable(torch.zeros(num_layers, batch_size, num_units)).type(dtype),
                 Variable(torch.zeros(num_layers, batch_size, num_units)).type(dtype))

class copy(nn.Module):
    def __init__(self, input_size, num_units, num_layers):
        self.input_size = input_size
        super(copy, self).__init__()
        self.lstm = nn.LSTM(input_size, num_units, num_layers, batch_first = True)
    def forward(self,input, initial_state = init):
        out = self.lstm(input, initial_state)
        return out
criterion = nn.BCELoss()
model = copy(seq_width + 1, num_units, num_layers)
model.cuda()
optimizer = torch.optim.RMSprop(model.parameters(), momentum = 0.9, alpha = 0.95, lr = 1e-4)
def clip_grads(model):
    parameters =list(filter(lambda p:p.grad is not None, model.parameters()))
    for p in parameters:
        p.grad.data.clamp_(-10, 10)
def train_batch(model, optimizer, X, Y):
    optimizer.zero_grad()
    output, final_state = model(X)
    input_zeros = Variable(torch.zeros(X.size(0), X.size(1)-1, X.size(2))).type(dtype)
    y_out, _= model(input_zeros, final_state)
    sigmoid_funcition = nn.Sigmoid()
    y_out = sigmoid_funcition(y_out)
    eps = 1e-8
    loss_data = -(Y*torch.log(y_out + eps) + (1 - Y)*torch.log(1 - y_out + eps)).mean()
    loss_data.backward()
    clip_grads(model)
    optimizer.step()
    y_out_binarized = y_out.clone().cpu().data
    y_out_binarized.apply_(lambda x:0 if x<0.5 else 1)
    cost = torch.sum(torch.abs(y_out_binarized - Y.cpu().data))
    return loss_data.data[0], cost/batch_size
def train_model(model):
    losses = []
    costs = []
    seq_lengths = []
    for batch_num, x, y in process_copy_data(num_batches, batch_size, seq_width, min_len, max_len):
        loss, cost = train_batch(model, optimizer, x, y)
        losses.append(loss)
        costs.append(cost)
        seq_lengths.append(y.size(0))
        if batch_num %200:
            mean_loss = np.array(losses[-interval:]).mean()
            mean_cost = np.array(costs[interval:]).mean()/y.size(1)
            logger.scalar_summary('loss', mean_loss, batch_num)
            logger.scalar_summary('mean_cost', mean_cost, batch_num)



if __name__ == '__main__':
    train_model(model)































































