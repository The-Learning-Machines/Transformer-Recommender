#!/usr/bin/env python36
# -*- coding: utf-8 -*-

######################################################
# Adapted from CRIPAC-DIG/SR-GNN for fair comparison #
######################################################
from einops import rearrange, repeat
import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.nn import TransformerEncoder
from torch.nn import TransformerEncoderLayer
from conformer import ConformerConvModule

class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, seq_dim = 1, offset = 0):
        t = torch.arange(x.shape[seq_dim], device = x.device).type_as(self.inv_freq) + offset
        sinusoid_inp = torch.einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb[None, :, :]


class Conformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout, kernel_size=31, causal=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                ConformerConvModule(
                    dim=dim,
                    # auto-regressive or not - 1d conv will be made causal with padding if so
                    causal=causal,
                    # what multiple of the dimension to expand for the depthwise convolution
                    expansion_factor=2,
                    kernel_size=kernel_size,           # kernel size, 17 - 31 was said to be optimal
                    dropout=dropout                # dropout at the very end
                )
            )

    def forward(self, x, mask=None):
        if mask is not None:
            raise(ValueError("No support for masks in Conformer yet!"))
        for layer in self.layers:
            x = layer(x) + x
        return x

class ConformerEncoder(nn.Module):
    def __init__(self, *, num_tokens, num_classes, dim, depth, heads, dim_head=64, dropout=0., emb_dropout=0., kernel_size=31, causal=False):
        super().__init__()

        self.item_embed = nn.Embedding(num_tokens, dim)
        self.pos_emb = FixedPositionalEmbedding(dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Conformer(
            dim, depth, heads, dim_head, dropout, kernel_size, causal)

        self.mlp = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim // 2),
                nn.ELU(inplace=True),
                nn.Linear(dim // 2, dim),
            )

    def forward(self, x, mask=None):

        b, n = x.shape
        pos_emb = self.pos_emb(x)
        x = pos_emb + self.item_embed(x)
    

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)

        x = self.transformer(x, mask)
        x = x.mean(dim=1)
        decoder_out = self.mlp(x)
        return decoder_out


class SelfAttentionNetwork(Module):
    def __init__(self, opt, n_node):
        super(SelfAttentionNetwork, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        # self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        # self.transformerEncoderLayer = TransformerEncoderLayer(d_model=self.hidden_size, nhead=opt.nhead,dim_feedforward=self.hidden_size * opt.feedforward)
        # self.transformerEncoder = TransformerEncoder(self.transformerEncoderLayer, opt.layer)
        self.transformerEncoder = ConformerEncoder(num_tokens=self.n_node, num_classes=self.n_node, dim=self.hidden_size, depth=opt.layer, heads=opt.nhead, dim_head=dim, dropout=0.1, emb_dropout=0.1, causal=False, kernel_size=3)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    # def compute_scores(self, hidden, mask):
    #     ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
    #     # print(hidden.shape, mask.shape, self.embedding.weight.shape)
    #     b = self.embedding.weight[1:]  # n_nodes x latent_size
    #     scores = torch.matmul(ht, b.transpose(1, 0))
    #     return scores
    # def forward(self, inputs, A):
    #     hidden = self.embedding(inputs)
    #     hidden = hidden.transpose(0,1).contiguous()
    #     hidden = self.transformerEncoder(hidden)
    #     hidden = hidden.transpose(0,1).contiguous()
    #     return hidden

    def compute_scores(self, hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        print(hidden.shape, mask.shape, self.embedding.weight.shape)
        b = self.transformerEncoder.item_embed.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(ht, b.transpose(1, 0))
        return scores


    def forward(self, inputs):
        hidden = self.transformerEncoder(inputs)
        # hidden = hidden.transpose(0,1).contiguous()
        hidden = torch.matmul(hidden, self.transformerEncoder.item_embed.weight[1:].transpose(1, 0))
        return hidden

        
def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    # alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    # A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    hidden = model(items)
    return targets, hidden


def train_test(model, train_data, test_data):    
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores = forward(model, i, test_data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    model.scheduler.step()
    return hit, mrr
