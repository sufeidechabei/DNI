import torch
import os
import numpy as np
from torch.autograd import Variable
num_steps = 20
ftype = torch.FloatTensor
dtype = torch.cuda.FloatTensor


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)


class Corpus(object):
    def __init__(self, path='./data'):
        self.dictionary = Dictionary()

    def get_data(self, path, batch_size=20):
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

                    # Tokenize the file content
        ids = torch.LongTensor(tokens)
        token = 0
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        return ids
def load_character_data(path):
        data = open(path, 'r').read()
        data = list(data)
        char_data = list(set(data))
        char_data = sorted(char_data)
        char_to_idx = {char:i for i, char in enumerate(char_data)}
        idx_to_char = {i:char for i, char in enumerate(char_data)}
        ids = []
        for char in data:
            ids.append(char_to_idx[char])
        load_data = torch.LongTensor(len(ids))
        for i in range(len(ids)):
            load_data[i] = ids[i]
        return load_data, char_to_idx, idx_to_char
def load_valid_character_data(path, char_to_idx):
    data = open(path, 'r').read()
    valid_data = list(data)
    valid_ids = []
    for char in valid_data:
        valid_ids.append(char_to_idx[char])
    idx = torch.LongTensor(len(valid_ids))
    for i in range(len(valid_ids)):
        idx[i] = valid_ids[i]
    return idx

def convert(data, unroll, num_steps):
    datalist = torch.split(data, 1, dim =1 )
    x0 = torch.cat(datalist[:unroll], dim=1)
    x1 = torch.cat(datalist[unroll:], dim=1)
    dataconvert = torch.cat((x1, x0), dim=1)
    return dataconvert
def character_data_expand(data, vocab):
    row = data.size()[0]
    col = data.size()[1]
    expand_data = torch.zeros(row,col,vocab)
    for i in range(row):
        for j in range(col):
            expand_data[i][j][int(data[i][j])] = 1
    return expand_data
def process_minist_data(minist, unrolling_step):
    datalist = torch.split(minist, 1 , dim=1)
    x0 = torch.cat(datalist[:unrolling_step], dim=1)
    x1 = torch.cat(datalist[unrolling_step:], dim=1)
    dataconvert = torch.cat((x1, x0), dim=1)
    return dataconvert
def process_copy_data(num_batches, batch_size, seq_width, min_len, max_len):
    for batch_num in range(num_batches):
        seq_len = np.random.randint(min_len, max_len)
        seq = np.random.randint(0, 2,(batch_size, seq_len, seq_width))
        seq = Variable(torch.from_numpy(seq)).type(dtype)
        inp = Variable(torch.zeros(batch_size, seq_len + 1, seq_width + 1)).type(dtype)
        inp[:, :seq_len, :seq_width] = seq.cpu().data.clone()
        inp[:, seq_len, seq_width] = 1.0
        outp = seq.clone().type(dtype)
        yield batch_num+1, inp, outp


def convert_copy_data(inp):
    convert_inp = inp.data.clone()
    datalist = torch.split(convert_inp, 1, dim = 1)
    x0 = torch.cat(datalist[:unroll], dim = 1)
    x1 = torch.cat(datalist[unroll:], dim = 1)
    dataconvert = torch.cat((x1, x0), dim =1)
    return dataconvert






if __name__ == '__main__':
    pass



