import torch
import matplotlib.pyplot as plt
import copy
from torch.autograd import Variable
from torch import nn
from data_utils import *
from syntheticmodel import *
from logger import Logger
plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
num_batches = 60000
num_layers = 1
batch_size = 16
unrolling_step = 3
hidden_size = 200
num_layers = 1
output_size = 8
seq_width = 8
min_len = 8
max_len = 9
dtype = torch.cuda.FloatTensor
ftype = torch.FloatTensor
logger = Logger('./dnicopytask')
class DNICopy(nn.Module):
    def __init__(self):
        super(DNICopy, self).__init__()
        self.lstm = nn.LSTM(seq_width + 1, hidden_size, num_layers, batch_first = True)
        self.linear = nn.Linear(hidden_size, output_size + 2*hidden_size)
        self.initial_state = None
        self.init_weights()
        self.prev_state = [0, 0]
    def init_weights(self):
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        for names in self.lstm._all_weights:
            for name in filter(lambda n:"bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start,end = n//4, n//2
                bias.data[start:end].fill_(0.)

    def forward(self, input):
        out, final_state = self.lstm(input, self.initial_state)
        output = self.linear(out)
        return output, final_state
model = DNICopy()
model.cuda()
optimizer = torch.optim.RMSprop(model.parameters(), momentum = 0.9, alpha = 0.95, lr = 1e-4)
sg_model_loss = nn.MSELoss()
def train_batch(model, x, y):
    x_convert = convert_copy_data(x, unrolling_step)
    model.initial_state = (Variable(torch.zeros(num_layers, batch_size, hidden_size).type(dtype), requires_grad = True),
                           Variable(torch.zeros(num_layers, batch_size, hidden_size).type(dtype), requires_grad = True))
    for step in range(int(max_len/unrolling_step - 1)):
        optimizer.zero_grad()
        input = Variable(x[:, step*unrolling_step:(step+1)*unrolling_step, :].cpu().data.clone()).type(dtype)
        next_input = Variable(x_convert[:, step*unrolling_step:(step+1)*unrolling_step, :].cpu().data.clone())\
            .type(dtype)
        model.prev_state = [model.initial_state[0], model.initial_state[1]]
        model.initial_state[0].requires_grad = True
        model.initial_state[1].requires_grad = True
        model.prev_state[0].requires_grad = True
        model.prev_state[1].requires_grad = True
        output, final_state = model(input)
        c_sg, h_sg = build_synthetic_gradient(output, output_size, hidden_size)
        final_first = final_state[0].detach()
        final_second = final_state[1].detach()
        final_first.requires_grad = True
        final_second.requires_grad = True
        final_result = torch.cat([final_state[0], final_state[1]], dim = 0)
        model.initial_state = (final_first, final_second)
        nextoutput, _ = model(next_input)
        c_next_sg, h_next_sg = build_next_synthetic_gradient(nextoutput, output_size, hidden_size)
        next_sg = torch.cat([h_next_sg, c_next_sg], dim=0)
        sg = torch.cat([h_sg, c_sg], dim=0)
        final_result.backward(next_sg, retain_graph = True)
        sg_target = torch.cat([model.prev_state[0].grad, model.prev_state[1].grad], dim=0)
        sg_loss = sg_model_loss(sg, sg_target.detach())
        sg_loss.backward()
        clip_grads(model)
        optimizer.step()
    final_step = step + 1
    input = Variable(x[:, final_step*unrolling_step:(final_step + 1)*unrolling_step, :].cpu().data.clone()).type(dtype)
    output, final_state = model(input)
    c_sg, h_sg = build_synthetic_gradient(output, output_size, hidden_size)
    optimizer.zero_grad()
    zero_input = Variable(torch.zeros(x.size(0), x.size(1) - 1, x.size(2)).type(dtype))
    model.initial_state = final_state
    model.prev_state = [model.initial_state[0], model.initial_state[1]]
    result, _= model(zero_input)
    result = result[:, :, :output_size]
    sigmoid_function = nn.Sigmoid()
    y_out = sigmoid_function(result)
    Y = y
    eps = 1e-8
    loss_data = -(Y * torch.log(y_out + eps) + (1 - Y) * torch.log(1 - y_out + eps)).mean()
    loss_data.backward(retain_graph = True)
    sg = torch.cat([h_sg, c_sg], dim=0)
    sg_target = torch.cat([model.initial_state[0], model.initial_state[1]], dim = 0)
    sg_loss = sg_model_loss(sg, sg_target.detach())
    sg_loss.backward()
    clip_grads(model)
    optimizer.step()
def train(model):
    losses = []
    steps = []
    best_model_loss = 500
    for batch, x, y in process_copy_data(num_batches, batch_size, seq_width, min_len, max_len):
        train_batch(model, x, y)
        if (batch+1)%1000 == 0:
            inp, outp = generate_testcopy_data(1, 2, seq_width = 8)
            model.initial_state = (Variable(torch.zeros(num_layers, 1, hidden_size).type(dtype)),
                                   Variable(torch.zeros(num_layers, 1, hidden_size).type(dtype)))
            _, final_state = model(inp)
            model.inital_state = final_state
            zeros_input = Variable(torch.zeros(outp.size(0), outp.size(1), outp.size(2) + 1).type(dtype))
            output, _ = model(zeros_input)
            output = output[:, :, :output_size]
            sigmoid_funcition = nn.Sigmoid()
            y_out = sigmoid_funcition(output)
            eps = 1e-8
            Y = outp
            loss_data = -(Y * torch.log(y_out + eps) + (1 - Y) * torch.log(1 - y_out + eps)).mean()
            losses.append(loss_data.cpu().data.numpy())
            steps.append(batch + 1)
            logger.scalar_summary('Validation loss', loss_data.cpu().data, batch+1)
            if float(loss_data.cpu().data) <= best_model_loss:
                best_model_loss = float(loss_data.cpu().data)
                torch.save(model.state_dict(), './ModelZoo/'+str(batch + 1)+'bestmodel'+
                           str(loss_data.cpu().data.numpy()[0])+'.ckpt')


    plt.figure()
    plt.plot(steps, losses)
    plt.xlabel('Steps')
    plt.ylabel('Validation Loss')
    plt.savefig('./dnicopytask.jpg')
def clip_grads(model):
    parameters  = list(filter(lambda p:p.grad is not None, model.parameters()))
    for p in parameters:
        p.grad.data.clamp_(-10, 10)
if __name__ == '__main__':
    train(model)



