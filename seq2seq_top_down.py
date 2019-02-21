# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 20:29:54 2018

@author: Administrator
"""
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import pickle

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data.dataset as Dataset
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 100

import os
import numpy
from PIL import Image
import re
import pickle

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        #TODO: I'm not sure if embedding works
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        #maybe delete embedding
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.leaky_relu(output)
        output, hidden = self.lstm(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

teacher_forcing_ratio = 1.0


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    #print(input_tensor.shape)
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[0]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            
            loss += criterion(decoder_output, target_tensor[di].reshape(-1))
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di].type(torch.cuda.LongTensor).reshape(-1))
            if decoder_input.item() == 0:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adadelta(encoder.parameters())
    decoder_optimizer = optim.Adadelta(decoder.parameters())
    training_pairs = [random.choice(train_data) for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1].type(torch.cuda.LongTensor)
        
        if input_tensor.size(0)>MAX_LENGTH:
            continue
        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % (print_every*10)==0:
            evaluateRandomly(encoder,decoder)

#Evaluation
def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = sentence
        input_length = input_tensor.size(0)
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item()==1:
                break
            decoded_words.append(topi.item())

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(dev_data)
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence=""
        for word in output_words:
            output_sentence+=str(word)+","
        print('<', output_sentence)
        print('')

if '__main__'==__name__:
    with open('deepbank-preprocessed/word_to_ix.pickle','rb') as f:
        word2index=pickle.load(f)

    with open('deepbank-preprocessed/word_list.pickle','rb') as f:
        index2word=pickle.load(f)

    with open('deepbank-preprocessed/rule_to_ix.pickle','rb') as f:
        rule2index=pickle.load(f)
    
    with open('deepbank-preprocessed/rule_list.pickle','rb') as f:
        index2rule=pickle.load(f)
    
    dataset=[]
    with open("deepbank-preprocessed/src.train") as f:
        t=0
        for line in f:
            if t==0:
                x=[]
                for word in line.strip().split("  "):
                    index=word2index[word]
                    x.append(index)
                x.append(1)
                x=torch.cuda.LongTensor(x)
            else:
                y=[]
				tree=[]
				stack=[]
				index=[]
                for word in line.strip().split("  "):
                    type=word.split("-")[0]
					if type=="SHIFT":
						tree.append([-1,-1,len(index)])
						stack.append(len(tree)-1)
						
					elif type=="UNARY":
						tree.append([-1,stack.pop(),len(index)])
						stack.append(len(tree)-1)

					elif type=="BINARY":
						tree.append([stack.pop(),stack.pop(),len(index)])
						stack.append(len(tree)-1)
					
					else:
						assert(len(stack)==1)

					index.append(int(word.split("-")[1]))
				
				stack=[tree[-1]]
				while(len(stack)>0):
					nodes=stack.pop()
					y.append(index[nodes[2]])
					if nodes[1]==-1:
						continue
					elif nodes[0]==-1:
						stack.append(tree[nodes[1]])
					else:
						stack.append(tree[nodes[0]])
						stack.append(tree[nodes[1]])

				y=torch.cuda.LongTensor(y)
                dataset.append((x,y))

            t=1-t

    tr=int(round(len(dataset)*0.9))
    dev=len(dataset)
    train_data=dataset[:tr]
    dev_data=dataset[tr:dev]

    EOS_token=1
    SOS_token=0

    print("START TRAINING")

    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
    dev_loader = DataLoader(dataset=dev_data, batch_size=64)

    hidden_size = 256
    encoder1 = EncoderRNN(len(index2word), hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, len(index2rule), dropout_p=0.1).to(device)

    trainIters(encoder1, attn_decoder1, 200000, print_every=1000)
    
    torch.save(encoder1,'encoder.pkl')
    torch.save(attn_decoder1,'decoder.pkl')
    
    '''encoder1=torch.load('encoder.pkl')
    attn_decoder1=torch.load('decoder.pkl')'''
    evaluateRandomly(encoder1, attn_decoder1)

