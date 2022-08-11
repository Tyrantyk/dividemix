import torch
import logging
import argparse
import logging
import math
import os
import random
import shutil
import time
from collections import OrderedDict
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from model.PreResNet import *

def estimator(index, logits_w, targets, proto_label, feat_pid, k):
    p = torch.softmax(logits_w, dim=1)
    w = torch.ones((1, len(index)), dtype=torch.float32).squeeze(0).cuda()

    for i, idx in enumerate(index):
        if proto_label[feat_pid[idx]] != targets[i]:
            if targets[i] in p[i].topk(k)[1]:
                w[i] = p[i][targets[i]]
            else:
                w[i] = 0
    return w

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0.001, math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def create_model(args, device):
    model = swav(args)
    model_dict = torch.load("./checkpoint/pretrain500.ckpt", map_location={'cuda:3':"cuda:0"})
    model_dict['state_dict']["classifier.bias"] = torch.randn((1, 10), dtype=torch.float32).squeeze()
    model_dict['state_dict']["classifier.weight"] = torch.randn((10, 128), dtype=torch.float32)
    model_dict['state_dict'].pop("queue")
    model.load_state_dict(model_dict['state_dict'])
    model = model.to(device)
    return model


