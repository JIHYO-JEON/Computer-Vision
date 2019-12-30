import os
import sys
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.autograd import Variable

# from tensorboardX import SummaryWriter as writer

from PA3.data import *
from PA3.utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR


def train(epoch):
    net.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        if epoch <= args.warm:
            warmup_scheduler.step()

        images = Variable(images)
        labels = Variable(labels)

        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar100_training_loader.dataset)
        ))

def eval_training(epoch):
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:
        images = Variable(images)
        labels = Variable(labels)

        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset)
    ))
    print()


    return correct.float() / len(cifar100_test_loader.dataset)


# --------------------------------------------       -------------------------------------------
# -------------------------------------------- Main -------------------------------------------
# --------------------------------------------       -------------------------------------------


class args():
    net = 'se_resnet' # resnet or se_resnet
    gpu = True
    w = 2 #workers
    b = 64 #batch size
    s = True #shuffle
    warm = 1 #warm up training phase
    lr = 0.1 #initial learning rate

net = get_network(args, use_gpu=args.gpu)

# data preprocessing:
cifar100_training_loader = get_training_dataloader(
    settings.CIFAR100_TRAIN_MEAN,
    settings.CIFAR100_TRAIN_STD,
    num_workers=args.w,
    batch_size=args.b,
    shuffle=args.s
)

cifar100_test_loader = get_test_dataloader(
    settings.CIFAR100_TEST_MEAN,
    settings.CIFAR100_TEST_STD,
    num_workers=args.w,
    batch_size=args.b,
    shuffle=args.s
)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES,
                                                 gamma=0.2)  # learning rate decay
iter_per_epoch = len(cifar100_training_loader)
warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net)

# create checkpoint folder to save model
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

best_acc = 0.0
for epoch in range(1, settings.EPOCH):
    if epoch > args.warm:
        train_scheduler.step(epoch)

    train(epoch)
    acc = eval_training(epoch)

    # start to save best performance model after learning rate decay to 0.01
    if epoch > settings.MILESTONES[1] and best_acc < acc:
        torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
        best_acc = acc
        continue

    if not epoch % settings.SAVE_EPOCH:
        torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))
