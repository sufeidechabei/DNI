import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from data_utils import *
from syntheticmodel import *
from torch.autograd import Variable
from logger import Logger
import os
ftype = torch.cuda.FloatTensor
dtype = torch.cuda.LongTensor
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logger = Logger('./withoutlocalgrad')
# Hyper-parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 50
learning_rate = 0.01
unrolling_step = 7

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, initial_state):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes + 4*hidden_size)
        self.init_weight()
        self.initial_state = initial_state
        self.prev_state =[0, 0]
    def init_weight(self):
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        for names in self.lstm._all_weights:
            for name in filter(lambda n:"bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start,end = n//4, n//2
                bias.data[start:end].fill_(0.)


    def forward(self, x):
        # Set initial hidden and cell states
        # Forward propagate LSTM
        out, final_state = self.lstm(x, tuple(self.initial_state))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out, final_state

initial_state = (Variable(torch.zeros(num_layers, batch_size, hidden_size)),
                 Variable(torch.zeros(num_layers, batch_size, hidden_size)))
model = RNN(input_size, hidden_size, num_layers, num_classes, initial_state)
model = model.cuda()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
sg_model_loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=7*10e-5)

# Train the model
total_step = len(train_loader)
model.train()
epoch_size = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        model.initial_state = [Variable(torch.zeros(num_layers, batch_size, hidden_size).type(ftype),
                                        requires_grad=True),
                               Variable(torch.zeros(num_layers, batch_size, hidden_size).type(ftype),
                                        requires_grad=True)]
        labels = Variable(labels).type(dtype)
        for j in range(sequence_length//unrolling_step):
            images = images.view(-1, sequence_length, input_size)
            next_images = process_minist_data(images, unrolling_step)
            input = images[:, j*unrolling_step:(j + 1)*unrolling_step, :]
            next_input = next_images[:, j*unrolling_step:(j + 1)*unrolling_step, :]
            input = Variable(input,requires_grad = False).type(ftype)
            next_input = Variable(next_input, requires_grad = False).type(ftype)
            model.prev_state[0] = model.initial_state[0]
            model.prev_state[1] = model.initial_state[1]
            model.initial_state[0].requires_grad = True
            model.initial_state[1].requires_grad = True
            model.prev_state[0].requires_grad = True
            model.prev_state[1].requires_grad = True
            output, final_state = model(input)
            model.initial_state[0] = final_state[0].detach()
            model.initial_state[1] = final_state[1].detach()
            if j!= sequence_length//unrolling_step - 1:
                next_output, _ =model(next_input)
                c_sg, h_sg = build_synthetic_gradient(output, num_classes, hidden_size)
                c_next_sg, h_next_sg = build_next_synthetic_gradient(next_output, num_classes, hidden_size)
                optimizer.zero_grad()
                h_initial, c_initial = model.prev_state
                h_final, c_final = final_state
                final_result = torch.cat([h_final, c_final], dim=0)
                next_sg = torch.cat([h_next_sg, c_next_sg], dim=0)
                final_result.backward(next_sg, retain_graph = True)
                sg_target = torch.cat([h_initial.grad, c_initial.grad], dim=0)
                sg = torch.cat([h_sg, c_sg], dim=0)
                sg_loss = sg_model_loss(sg, sg_target.detach())
                sg_loss.backward()
                optimizer.step()
            else:
                next_output, _ = model(next_input)
                c_sg, h_sg = build_synthetic_gradient(output, num_classes, hidden_size)
                c_next_sg, h_next_sg = build_next_synthetic_gradient(next_output, num_classes, hidden_size)
                optimizer.zero_grad()
                h_final, c_final = final_state
                final_result = torch.cat([h_final, c_final], dim=0)
                next_sg = torch.cat([h_next_sg, c_next_sg], dim=0)
                next_sg.data = torch.zeros(next_sg.size()).type(ftype)
                final_result.backward(next_sg, retain_graph = True)
                slice_output = output[:,-1,0:num_classes]
                loss = criterion(slice_output, labels)
                loss.backward(retain_graph = True)
                begin_data = h_initial.grad.data
                h_initial, c_initial = model.prev_state
                sg_target =torch.cat((h_initial.grad, c_initial.grad), dim = 0)
                sg = torch.cat([h_sg, c_sg], dim=0)
                sg_loss1= sg_model_loss(sg, sg_target.detach())
                sg_loss1.backward()
                end_data = h_initial.grad.data
                optimizer.step()
        count = i+epoch*epoch_size
        if count%10==0:
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, to_np(value), count)
                logger.histo_summary(tag + '/grad', to_np(value.grad), count)
            logger.scalar_summary('loss', loss.data, count)
            logger.scalar_summary('sg_loss', sg_loss.data, count)
            print ("when the iteration is " + str(count) + " the loss is " +str(loss.data))


