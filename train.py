from __future__ import print_function
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn

from torch.utils.data.dataloader import DataLoader
import numpy as np
from model import FBSD
from datesets import get_trainAndtest

from config import class_nums
from config import HyperParams

def train():
    # output dir
    output_dir = HyperParams['kind']+ '_'+ HyperParams['arch']+ '_output'
    try:
        os.stat(output_dir)
    except:
        os.makedirs(output_dir)
    # Data
    trainset, testset = get_trainAndtest()
    trainloader = DataLoader(trainset, batch_size=HyperParams['bs'], shuffle=True, num_workers=8, pin_memory=True)
    testloader = DataLoader(testset, batch_size=HyperParams['bs'], shuffle=False, num_workers=8)

    ####################################################
    print("dataset: ", HyperParams['kind'])
    print("backbone: ", HyperParams['arch'])
    print("trainset: ", len(trainset))
    print("testset: ", len(testset))
    print("classnum: ", class_nums[HyperParams['kind']])
    ####################################################

    net = FBSD(class_num=class_nums[HyperParams['kind']], arch=HyperParams['arch'])
    net = net.cuda()
    netp = nn.DataParallel(net).cuda()

    CELoss = nn.CrossEntropyLoss()

    ########################
    new_params, old_params = net.get_params()
    new_layers_optimizer = optim.SGD(new_params, momentum=0.9, weight_decay=5e-4, lr=0.002)
    old_layers_optimizer = optim.SGD(old_params, momentum=0.9, weight_decay=5e-4, lr=0.0002)
    new_layers_optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(new_layers_optimizer, HyperParams['epoch'], 0)
    old_layers_optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(old_layers_optimizer, HyperParams['epoch'], 0)

    max_val_acc = 0
    for epoch in range(0, HyperParams['epoch']):
        print('\nEpoch: %d' % epoch)
        start_time = datetime.now()
        print("start time: ", start_time.strftime('%Y-%m-%d-%H:%M:%S'))
        net.train()
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        train_loss4 = 0
        correct = 0
        total = 0
        idx = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            idx = batch_idx
            inputs, targets = inputs, targets.cuda()
            output_1, output_2, output_3, output_concat = netp(inputs)

            # adjust optimizer lr
            new_layers_optimizer_scheduler.step()
            old_layers_optimizer_scheduler.step()

            # overall update
            loss1 = CELoss(output_1, targets)*2
            loss2 = CELoss(output_2, targets)*2
            loss3 = CELoss(output_3, targets)*2
            concat_loss = CELoss(output_concat, targets)

            new_layers_optimizer.zero_grad()
            old_layers_optimizer.zero_grad()
            loss = loss1 + loss2 + loss3 + concat_loss
            loss.backward()
            new_layers_optimizer.step()
            old_layers_optimizer.step()

            #  training log
            _, predicted = torch.max((output_1+output_2+output_3+output_concat).data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            train_loss += (loss1.item() + loss2.item() + loss3.item() + concat_loss.item() )
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_loss3 += loss3.item()
            train_loss4 += concat_loss.item()

            if batch_idx % 50 == 0:
                print('Step: %d | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss_concat: %.5f | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    batch_idx, train_loss1 / (batch_idx + 1), train_loss2 / (batch_idx + 1),
                    train_loss3 / (batch_idx + 1), train_loss4 / (batch_idx + 1), train_loss / (batch_idx + 1),
                    100. * float(correct) / total, correct, total))
        train_acc = 100. * float(correct) / total
        train_loss = train_loss / (idx + 1)
        # eval
        val_acc = test(net, testloader)
        torch.save(net.state_dict(), './' + output_dir + '/current_model.pth')
        if val_acc  > max_val_acc:
            max_val_acc = val_acc 
            torch.save(net.state_dict(), './' + output_dir + '/best_model.pth')
        print("best result: ", max_val_acc)
        print("current result: ", val_acc)
        end_time = datetime.now()
        print("end time: ", end_time.strftime('%Y-%m-%d-%H:%M:%S'))

def test(net, testloader):
    net.eval()
    correct_com = 0
    total = 0

    softmax = nn.Softmax(dim=-1)

    for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                output_1, output_2, output_3, output_concat = net(inputs)
                outputs_com = output_1 + output_2 + output_3 + output_concat

            _, predicted_com = torch.max(outputs_com.data, 1)
            total += targets.size(0)
            correct_com += predicted_com.eq(targets.data).cpu().sum()
    test_acc_com = 100. * float(correct_com) / total

    return test_acc_com 

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

import os
if __name__ == '__main__':
    set_seed(666)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    train()
