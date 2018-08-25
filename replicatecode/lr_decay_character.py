import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from data_utils import *
from torch.nn import Dropout
from syntheticmodel import *
from logger import Logger
from torch import nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Hyper Parameters

hidden_size = 1024
num_layers = 2
num_epochs = 200
num_samples = 1000  # number of words to be sampled
batch_size = 20
seq_length = 20
learning_rate = 0.002
num_steps = 20
unrolling_size = 5
dtype = torch.cuda.LongTensor
ftype = torch.cuda.FloatTensor
import copy


# Load Penn Treebank Dataset
train_path = '/data/zhanghao/must/DNIRNNReplicate/data/ptb.char.train.txt'
valid_path = '/data/zhanghao/must/DNIRNNReplicate/data/ptb.char.valid.txt'
lr = 7*10e-5

raw_data, char_to_idx, idx_to_char  = load_character_data(train_path)
valid_data = load_valid_character_data(valid_path, char_to_idx)
vocab_size = len(char_to_idx)
data_len = len(raw_data)
valid_data_len = len(valid_data)
n_seq = (data_len - 1)//num_steps
n_valid_seq = (valid_data_len - 1)//num_steps
raw_data_x = raw_data[0:n_seq*num_steps].view(n_seq,num_steps)
raw_data_x = raw_data_x.type(ftype)
valid_raw_data_x = valid_data[0:n_valid_seq*num_steps].view(n_valid_seq, num_steps).type(ftype)
valid_raw_data_y = valid_data[1:n_valid_seq*num_steps + 1].view(n_valid_seq, num_steps).type(dtype)
valid_raw_data_x = character_data_expand(valid_raw_data_x, vocab_size)
next_raw_data_x = convert(raw_data_x, 5, 20)
next_raw_data_x =next_raw_data_x.type(ftype)
raw_data_y = raw_data[1:n_seq*num_steps + 1].view(n_seq,num_steps)
raw_data_x = character_data_expand(raw_data_x, vocab_size)
next_raw_data_x = character_data_expand(next_raw_data_x, vocab_size)
logger = Logger('./lr_logs')
next_raw_data_x = convert(raw_data_x, 5, 20)
valid_epoch_size = valid_raw_data_x.size()[0] // batch_size

# RNN Based Language Model
class RNNLM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers,initial_state):
        super(RNNLM, self).__init__()
        self.lstm = nn.LSTM(vocab_size, hidden_size, num_layers, batch_first=True,dropout = 0.5)
        self.linear = nn.Linear(hidden_size, vocab_size + 4*hidden_size)
        self.drop = nn.Dropout(p = 0.5)
        self.init_weights()
        self.initial_state = initial_state
        self.prev_state = [0,0]
    def init_weights(self):
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        for names in self.lstm._all_weights:
            for name in filter(lambda n:"bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start,end = n//4, n//2
                bias.data[start:end].fill_(0.)

    def forward(self, x):
        # Forward propagate RNN
        out, final_state = self.lstm(x, tuple(self.initial_state))
        out = self.drop(out)
        out = self.linear(out)
        return out, final_state
initial_state = (Variable(torch.zeros(num_layers, batch_size, hidden_size)).type(ftype),
                 Variable(torch.zeros(num_layers, batch_size, hidden_size)).type(ftype))
def exp_lr_scheduler(optimizer, epoch,init_lr =lr, lr_decay_epoch = 10):
    lr = init_lr*(0.1**(epoch//lr_decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


model = RNNLM(vocab_size,  hidden_size, num_layers, initial_state)
model = model.cuda()
copymodel = None
base_model_loss = nn.CrossEntropyLoss()
sg_model_loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
best_model_bpc = 100.0
def train_model():
    model.train()
    for epoch in range(num_epochs):
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
                input_x = Variable(input_x, requires_grad = False).type(ftype)
                next_input_x = Variable(next_input_x, requires_grad = False).type(ftype)
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
                    slice_output = output[:,:,0:vocab_size]
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
                    slice_output = output[:, :, 0:vocab_size]
                    output_linear = slice_output.contiguous().view(output.size(0) * output.size(1), -1)
                    loss = base_model_loss(output_linear,label_y.view(-1))
                    loss.backward(retain_graph = True)
                    h_initial, c_initial = model.prev_state
                    sg_target = torch.cat([h_initial.grad, c_initial.grad], dim = 0)
                    sg = torch.cat([h_sg, c_sg], dim = 0)
                    sg_loss = sg_model_loss(sg, sg_target.detach())
                    sg_loss.backward()
                    optimizer.step()
                step = ((batch + 1)*4 + epoch*data_len//batch_size//unrolling_size)
                if step%100 == 0 :
                    model.eval()
                    BPC = 0
                    for batch in range(valid_epoch_size):
                        model.initial_state = [Variable(torch.zeros(num_layers, batch_size, hidden_size).type(ftype),
                                                        requires_grad=True),
                                               Variable(torch.zeros(num_layers, batch_size, hidden_size).type(ftype),
                                                        requires_grad=True)]
                        valid_input_x = Variable(valid_raw_data_x[batch * batch_size:(batch + 1) * batch_size, :]).type(
                            ftype)
                        valid_label_y = Variable(raw_data_y[batch * batch_size:(batch + 1) * batch_size, :]).type(dtype)
                        output, _ = model(valid_input_x)
                        result = output[:, :, :vocab_size]
                        result = result.contiguous().view(result.size(0) * result.size(1), - 1)
                        result_to_numpy = result.data.cpu().numpy()
                        target_to_numpy = valid_label_y.data.cpu().numpy()
                        prob_result = np.exp(result_to_numpy)
                        add_prob_result = np.sum(prob_result, axis=1)
                        prob_shape = prob_result.shape[1]
                        final_prob_result = prob_result / add_prob_result.repeat(prob_shape).reshape(-1, prob_shape)
                        target_to_numpy = target_to_numpy.reshape(-1)
                        row_index = np.arange(len(final_prob_result))
                        BPC = BPC + np.mean(-np.log2(final_prob_result[row_index, target_to_numpy]))
                    BPC = BPC / valid_epoch_size
                    logger.scalar_summary('BPC', BPC, step)
                    global best_model_bpc
                    if BPC < best_model_bpc:
                        best_model_bpc = BPC
                        copymodel = copy.deepcopy(model)
                    model.train()

        logger.scalar_summary('loss', to_np(loss), epoch)
        logger.scalar_summary('sg_loss', to_np(sg_loss), epoch)
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, to_np(value), epoch)
            logger.histo_summary(tag + '/grad', to_np(value.grad), epoch)
        print("The smallest value of BPC is" + str(best_model_bpc))
        optimizer = exp_lr_scheduler(optimizer, epoch)






if __name__ == '__main__':
     train_model()