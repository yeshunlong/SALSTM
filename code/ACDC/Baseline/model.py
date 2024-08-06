#!/usr/bin/env python
# -*- coding:utf-8 -*-

import math
import numpy as np
import torch
from torch import nn

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, query_dim):
        super(Attention, self).__init__()
        self.scale = 1. / math.sqrt(query_dim)

    def forward(self, query, keys, values):
        # Query = (B, T, Q)
        # Keys = (B, T, K)
        # Values = (B, T, V)
        # Outputs = lin_comb:(B, T, V)

        # Here we assume Q == K (dot product attention)
        keys = keys.transpose(1, 2)  # (B, T, K) -> (B, K, T)
        energy = torch.bmm(query, keys)  # (B, T, Q) x (B, K, T) -> (B, T, T)
        energy = F.softmax(energy.mul_(self.scale), dim=2)  # scale, normalize
        linear_combination = torch.bmm(energy, values)  # (B, T, T) x (B, T, V) -> (B, T, V)
        return linear_combination
    
class GatedAttention(nn.Module):
    def __init__(self, query_dim, nchannels):
        super(GatedAttention, self).__init__()
        self.scale = 1. / math.sqrt(query_dim)
        self.q_matrix = nn.Linear(nchannels, nchannels)
        self.k_matrix = nn.Linear(nchannels, nchannels)
        self.v_matrix = nn.Linear(nchannels, nchannels)

    def forward(self, x, g_query, g_keys, g_values):
        # x = (B, T, Q)
        # g_query, g_keys, g_values = (B, T, Q)
        # Outputs = lin_comb:(B, T, Q)
        x = x.transpose(1, 2) # (B, T, Q) -> (B, Q, T)
        g_query, g_keys, g_values = g_query.transpose(1, 2), g_keys.transpose(1, 2), g_values.transpose(1, 2) # (B, T, Q) -> (B, Q, T)
        query, keys, values = self.q_matrix(x), self.k_matrix(x), self.v_matrix(x) # query, keys, values = (B, Q, T)
        query, keys = g_query * query, g_keys * keys
        query = query.transpose(1, 2)  # (B, Q, T) -> (B, T, Q)
        energy = torch.bmm(query, keys)  # (B, T, Q) x (B, Q, T) -> (B, T, T)
        energy = F.softmax(energy.mul_(self.scale), dim=2)  # scale, normalize
        values = g_values * values
        values = values.transpose(1, 2)  # (B, Q, T) -> (B, T, Q)
        linear_combination = torch.bmm(energy, values)  # (B, T, T) x (B, T, Q) -> (B, T, Q)
        return linear_combination


class LSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels,
                 kernel_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.kernel_size = kernel_size
        self.num_features = 4
        self.padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(self.input_channels + self.hidden_channels, self.num_features * self.hidden_channels, self.kernel_size, 1, self.padding)

    def forward(self, x, h, c):
        combined = torch.cat((x, h), dim=1)
        A = self.conv(combined)

        # NOTE: A? = xz * Wx? + hz-1 * Wh? + b? where * is convolution
        (Ai, Af, Ao, Ag) = torch.split(A, A.size()[1] // self.num_features, dim=1)

        i = torch.sigmoid(Ai)     # input gate
        f = torch.sigmoid(Af)     # forget gate
        o = torch.sigmoid(Ao)     # output gate
        g = torch.tanh(Ag)

        c = c * f + i * g           # cell activation state
        h = o * torch.tanh(c)     # cell hidden state

        return h, c, i, f, o, g

    @staticmethod
    def init_hidden(batch_size, hidden_c, shape):
        try:
            return (Variable(torch.zeros(batch_size, hidden_c, shape[0], shape[1])).cuda(),
                    Variable(torch.zeros(batch_size, hidden_c, shape[0], shape[1])).cuda())
        except:
            return (Variable(torch.zeros(batch_size, hidden_c, shape[0], shape[1])),
                    Variable(torch.zeros(batch_size, hidden_c, shape[0], shape[1])))


class TransLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels,
                 kernel_size, bias, attention_size):
        super(TransLSTM, self).__init__()
        self.attention = Attention(attention_size)
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.bias = bias
        self.all_layers = []

        for layer in range(self.num_layers):
            name = 'cell{}'.format(layer)
            cell = LSTMCell(self.input_channels[layer], self.hidden_channels[layer], self.kernel_size, self.bias)
            setattr(self, name, cell)
            self.all_layers.append(cell)

    def forward(self, x):
        bsize, steps, _, height, width = x.size()
        internal_state = []
        outputs = []
        for step in range(steps):
            input = torch.squeeze(x[:, step, :, :, :], dim=1)
            for layer in range(self.num_layers):
                if step == 0:
                    (h, c) = LSTMCell.init_hidden(bsize, self.hidden_channels[layer], (height, width))
                    internal_state.append((h, c))
                name = 'cell{}'.format(layer)
                (h, c) = internal_state[layer]
                input, c, i, f, o, g = getattr(self, name)(input, h, c)  # forward propogation call
                internal_state[layer] = (input, c)
            outputs.append(input)
        output = outputs[-1]
        query, keys, values = self.get_QKV(output, c, i, f, g, o)
        query, keys, values = query.view(bsize, -1, height * width), keys.view(bsize, -1, height * width), values.view(bsize, -1, height * width)
        output = self.attention(query, keys, values)
        output = output.view(bsize, -1, height, width)
        return output
    
    def get_QKV(self, h_states, c_states, i_states, f_states, g_states, o_states):
        values = h_states
        query, keys = (c_states + f_states) / 2, (g_states + h_states) / 2
        return query, keys, values
    
    
class GatedTransLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels,
                 kernel_size, bias, attention_size):
        super(GatedTransLSTM, self).__init__()
        self.attention = GatedAttention(attention_size, input_channels)
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)

        self.bias = bias
        self.all_layers = []
        for layer in range(self.num_layers):
            name = 'cell{}'.format(layer)
            cell = LSTMCell(self.input_channels[layer], self.hidden_channels[layer], self.kernel_size, self.bias)
            setattr(self, name, cell)
            self.all_layers.append(cell)

    def forward(self, x):
        bsize, steps, _, height, width = x.size()
        internal_state = []
        outputs = []
        for step in range(steps):
            input = torch.squeeze(x[:, step, :, :, :], dim=1)
            for layer in range(self.num_layers):
                if step == 0:
                    (h, c) = LSTMCell.init_hidden(bsize, self.hidden_channels[layer], (height, width))
                    internal_state.append((h, c))
                name = 'cell{}'.format(layer)
                (h, c) = internal_state[layer]
                input, c, i, f, o, g = getattr(self, name)(input, h, c)  # forward propogation call
                internal_state[layer] = (input, c)
            outputs.append(input)
        output = outputs[-1]
        query, keys, values = self.get_QKV(output, c, i, f, g, o)
        output, query, keys, values = output.view(bsize, -1, height * width), query.view(bsize, -1, height * width), keys.view(bsize, -1, height * width), values.view(bsize, -1, height * width)
        output = self.attention(output, query, keys, values)
        output = output.view(bsize, -1, height, width)
        return output
    
    def get_QKV(self, h_states, c_states, i_states, f_states, g_states, o_states):
        values = h_states
        query, keys = (c_states + f_states) / 2, (g_states + h_states) / 2
        return query, keys, values
    
class TransLSTMLayer(nn.Module):
    # Constructor
    def __init__(self, input_channels, hidden_channels,
                 kernel_size, bias, num_classes, attenion_size):

        super(TransLSTMLayer, self).__init__()
        self.forward_net = TransLSTM(input_channels, hidden_channels, kernel_size, bias, attention_size=attenion_size)
        self.reverse_net = TransLSTM(input_channels, hidden_channels, kernel_size, bias, attention_size=attenion_size)
        self.conv = nn.Conv2d(2 * hidden_channels[-1], num_classes, kernel_size=1)

    def forward(self, x1, x2, x3):
        x1 = torch.unsqueeze(x1, dim=1)
        x2 = torch.unsqueeze(x2, dim=1)
        x3 = torch.unsqueeze(x3, dim=1)

        xforward = torch.cat((x1, x2), dim=1)
        xreverse = torch.cat((x3, x2), dim=1)

        yforward = self.forward_net(xforward)
        yreverse = self.reverse_net(xreverse)

        ycat = torch.cat((yforward, yreverse), dim=1)
        y = self.conv(ycat)
        return y


class SALSTM(nn.Module):
    def __init__(self, input_channels=64, hidden_channels=[64], kernel_size=5, bias=True, num_classes=4, attenion_size=224*224, encoder=None):
        super(SALSTM, self).__init__()
        self.encoder = encoder
        self.lstmlayer = TransLSTMLayer(input_channels, hidden_channels, kernel_size, bias, num_classes, attenion_size)

    def forward(self, x1, x2, x3):
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)
        x3 = self.encoder(x3)
        y = self.lstmlayer(x1, x2, x3)
        return y
    
    
class GatedTransLSTMLayer(nn.Module):
    # Constructor
    def __init__(self, input_channels, hidden_channels,
                 kernel_size, bias, num_classes, attenion_size):

        super(GatedTransLSTMLayer, self).__init__()
        self.forward_net = GatedTransLSTM(input_channels, hidden_channels, kernel_size, bias, attention_size=attenion_size)
        self.reverse_net = GatedTransLSTM(input_channels, hidden_channels, kernel_size, bias, attention_size=attenion_size)
        self.conv = nn.Conv2d(2 * hidden_channels[-1], num_classes, kernel_size=1)

    def forward(self, x1, x2, x3):
        x1 = torch.unsqueeze(x1, dim=1)
        x2 = torch.unsqueeze(x2, dim=1)
        x3 = torch.unsqueeze(x3, dim=1)

        xforward = torch.cat((x1, x2), dim=1)
        xreverse = torch.cat((x3, x2), dim=1)

        yforward = self.forward_net(xforward)
        yreverse = self.reverse_net(xreverse)

        ycat = torch.cat((yforward, yreverse), dim=1)
        y = self.conv(ycat)
        return y


class LSTMSA(nn.Module):
    def __init__(self, input_channels=64, hidden_channels=[64], kernel_size=5, bias=True, num_classes=4, attenion_size=224*224, encoder=None):
        super(LSTMSA, self).__init__()
        self.encoder = encoder
        self.lstmlayer = GatedTransLSTMLayer(input_channels, hidden_channels, kernel_size, bias, num_classes, attenion_size)

    def forward(self, x1, x2, x3):
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)
        x3 = self.encoder(x3)
        y = self.lstmlayer(x1, x2, x3)
        return y