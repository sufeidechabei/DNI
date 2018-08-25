import torch
from torch.autograd import Variable
def build_synthetic_gradient(out,vocab_size,hidden_size):
    out1 = out[:,0,vocab_size:]
    sg_gradient = torch.split(out1, hidden_size, dim=1)
    print(sg_gradient[0].size())
    c = []
    h = []
    for cell in range(0,len(sg_gradient),2):
        c.append(sg_gradient[cell].unsqueeze(dim = 0))
    for hidden in range(1,len(sg_gradient),2):
        h.append(sg_gradient[hidden].unsqueeze(dim = 0))
    print(c[0].size())
    cell_synthetic_gradient = torch.cat(c,dim = 0)
    hidden_synthetic_gradient = torch.cat(h,dim = 0)
    print(cell_synthetic_gradient.size())
    return cell_synthetic_gradient, hidden_synthetic_gradient
def build_next_synthetic_gradient(next_out,vocab_size,hidden_size):
    next_out1 = next_out[:,0,vocab_size:].unsqueeze(dim = 1)
    next_sg_gradient = torch.split(next_out1, hidden_size, dim=2)
    c = []
    h = []
    for cell in range(0,len(next_sg_gradient),2):
        c.append(next_sg_gradient[cell].unsqueeze(dim = 0))
    for hidden in range(1,len(next_sg_gradient),2):
        h.append(next_sg_gradient[hidden].unsqueeze(dim = 0))
    cell_next_synthetic_gradient = torch.cat(c,dim = 0)
    hidden_next_synthetic_gradient = torch.cat(h,dim = 0)
    return cell_next_synthetic_gradient,hidden_next_synthetic_gradient
a = torch.rand(20, 5 ,10800)
c = build_synthetic_gradient(a, 10000, 200)
