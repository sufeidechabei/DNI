import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from data_utils import Dictionary, Corpus
from torch.nn import Dropout
from syntheticmodel import *
from logger import Logger

from torch import nn

# Hyper Parameters
embed_size = 200
hidden_size = 200
num_layers = 2
num_epochs = 20
num_samples = 1000  # number of words to be sampled
batch_size = 20
seq_length = 20
learning_rate = 0.002
num_steps = 20
unrolling_size = 5
dtype = torch.cuda.LongTensor
ftype = torch.cuda.FloatTensor


# Load Penn Treebank Dataset
train_path = './data/ptb.train.txt'
test_path = '/data/ptb.test.txt'
corpus = Corpus('./data/ptb.train.txt')
raw_data = corpus.get_data(train_path, batch_size)
vocab_size = len(corpus.dictionary)
data_len = len(raw_data)
n_seq = (data_len - 1)//num_steps
raw_data_x = raw_data[0:n_seq*num_steps].view(n_seq,num_steps)
raw_data_y = raw_data[1:n_seq*num_steps + 1].view(n_seq,num_steps)
logger = Logger('./logs')

def convert(data,unroll,num_steps):
    datalist = torch.split(data, 1, dim =1 )
    x0 = torch.cat(datalist[:unroll], dim=1)
    x1 = torch.cat(datalist[unroll:], dim=1)
    dataconvert = torch.cat((x1, x0), dim=1)
    return dataconvert
next_raw_data_x = convert(raw_data_x, 5, 20)
next_raw_data_y = convert(raw_data_y, 5, 20)
# RNN Based Language Model
class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers,initial_state):
        super(RNNLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True,dropout = 0.5)
        self.linear = nn.Linear(hidden_size, vocab_size + 4*hidden_size)
        self.drop1 = nn.Dropout(p = 0.5)
        self.drop2 = nn.Dropout(p = 0.5)
        self.init_weights()
        self.initial_state = initial_state
        self.prev_state = [0,0]
    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        for names in self.lstm._all_weights:
            for name in filter(lambda n:"bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start,end = n//4, n//2
                bias.data[start:end].fill_(0.)

    def forward(self, x):
        # Embed word ids to vectors
        x = self.embed(x)
        x = self.drop1(x)
        # Forward propagate RNN
        out, final_state = self.lstm(x, tuple(self.initial_state))
        # Reshape output to (batch_size*sequence_length, hidden_size)
        out = out.contiguous()
       #Decode hidden states of all time step
        out = self.drop2(out)
        out = self.linear(out)
        return out, final_state
initial_state = (Variable(torch.zeros(num_layers,batch_size,hidden_size)).type(ftype),
                 Variable(torch.zeros(num_layers,batch_size,hidden_size)).type(ftype))


model = RNNLM(vocab_size, embed_size, hidden_size, num_layers,initial_state)
model = model.cuda()
base_model_loss = nn.CrossEntropyLoss()
sg_model_loss = nn.MSELoss()
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.1, lr_decay = 0.9)
def train_model():
    model.train()
    for i in range(num_epochs):
        for batch in range(raw_data_x.size()[0] // batch_size):
            model.initial_state = [Variable(torch.zeros(num_layers, batch_size, hidden_size).type(ftype),
                                            requires_grad = True),
                                   Variable(torch.zeros(num_layers, batch_size, hidden_size).type(ftype),
                                            requires_grad = True)]
            for j in range(seq_length // unrolling_size):
                input_x = raw_data_x[batch * batch_size:(batch + 1) * batch_size,
                           j * unrolling_size:(j + 1) * unrolling_size]
                next_input_x = next_raw_data_x[batch * batch_size:(batch + 1) * batch_size,
                                j * unrolling_size:(j + 1) * unrolling_size]
                label_y = raw_data_y[batch * batch_size:(batch + 1) * batch_size,
                          j * unrolling_size:(j + 1) * unrolling_size]
                input_x = Variable(input_x, requires_grad = False).type(dtype)
                next_input_x = Variable(next_input_x, requires_grad = False).type(dtype)
                label_y = Variable(label_y, requires_grad = False).type(dtype)
                model.prev_state[0] = model.initial_state[0]
                model.prev_state[1] = model.initial_state[1]
                model.prev_state[0].requires_grad = True
                model.prev_state[1].requires_grad = True
                model.initial_state[0].requires_grad = True
                model.initial_state[1].requires_grad = True
                output, final_state = model(input_x)
                model.initial_state[0] = final_state[0].detach()
                model.initial_state[1] = final_state[1].detach()
                if j != seq_length//unrolling_size - 1:
                    next_output, _ = model(next_input_x)
                    c_sg, h_sg = build_synthetic_gradient(output, vocab_size, hidden_size)
                    c_next_sg, h_next_sg  = build_next_synthetic_gradient(next_output, vocab_size, hidden_size)
                    global optimizer
                    optimizer.zero_grad()
                    h_initial, c_initial = model.prev_state
                    h_final, c_final = final_state
                    final_result = torch.cat([h_final,c_final], dim = 0)
                    next_sg = torch.cat([h_next_sg,c_next_sg], dim = 0)
                    final_result.backward(next_sg)
                    slice_output = output[:,:,0:10000]
                    output_linear = slice_output.contiguous().view(output.size(0)*output.size(1) , -1)
                    loss = base_model_loss(input = output_linear,target = label_y.view(-1))
                    loss.backward(retain_graph = True)
                    sg_target = torch.cat([h_initial.grad, c_initial.grad], dim = 0)
                    sg = torch.cat([h_sg, c_sg], dim = 0)
                    sg_loss = sg_model_loss(sg, sg_target.detach())
                    sg_loss.backward()
                    optimizer.step()
                else:
                    next_output, _ = model(next_input_x)
                    c_sg, h_sg = build_synthetic_gradient(output, vocab_size, hidden_size)
                    c_next_sg, h_next_sg  = build_next_synthetic_gradient(next_output, vocab_size, hidden_size)
                    optimizer.zero_grad()
                    h_final, c_final = final_state
                    final_result = torch.cat([h_final,c_final], dim = 0)
                    next_sg = torch.cat([h_next_sg,c_next_sg], dim = 0)
                    next_sg.data = torch.zeros(next_sg.size()).type(ftype)
                    final_result.backward(next_sg)
                    slice_output = output[:, :, 0:10000]
                    output_linear = slice_output.contiguous().view(output.size(0) * output.size(1), -1)
                    loss = base_model_loss(output_linear,label_y.view(-1))
                    loss.backward(retain_graph = True)
                    h_initial, c_initial = model.prev_state
                    sg_target = torch.cat([h_initial.grad, c_initial.grad], dim = 0)
                    sg = torch.cat([h_sg, c_sg], dim = 0)
                    sg_loss = sg_model_loss(sg, sg_target.detach())
                    sg_loss.backward()
                    optimizer.step()
                    logger.scalar_summary('loss', loss.data, i)
            step = ((batch + 1)*4 + i*(data_len)//batch_size//unrolling_size)
            if step%100 == 0 :
                logger.scalar_summary('loss', to_np(loss), step)
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, to_np(value), step)
                    logger.histo_summary(tag + '/grad', to_np(value.grad), step)






if __name__ == '__main__':
     train_model()

                    


                    

