import argparse
#from dataset import *

#from skimage import io
from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F

from PA3.data import *
from PA3.utils import get_network, get_test_dataloader



class args():
    net = 'resnet' # resnet or se_resnet
    weights = './checkpoint/resnet/resnet-190-regular.pth'
    gpu = True
    w = 2 #workers
    b = 64 #batch size
    s = True #shuffle
    warm = 1 #warm up training phase

args = args()
net = get_network(args)

cifar100_test_loader = get_test_dataloader(
    settings.CIFAR100_TRAIN_MEAN,
    settings.CIFAR100_TRAIN_STD,
    num_workers=args.w,
    batch_size=args.b,
    shuffle=args.s
)

net.load_state_dict(torch.load(args.weights), args.gpu)
print(net)
net.eval()

correct_1 = 0.0
correct_5 = 0.0
total = 0
for n_iter, (image, label) in enumerate(cifar100_test_loader):
    print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))
    image = Variable(image).cuda()
    label = Variable(label).cuda()
    output = net(image)
    _, pred = output.topk(5, 1, largest=True, sorted=True)

    label = label.view(label.size(0), -1).expand_as(pred)
    #test_loss += F.nll_loss(output, label, reduction='sum').item()
    correct = pred.eq(label).float()

    #compute top 5
    correct_5 += correct[:, :5].sum()

    #compute top1
    correct_1 += correct[:, :1].sum()

print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        total, correct_1, len(cifar100_test_loader.dataset),
        100. * correct_1 / len(cifar100_test_loader.dataset)))
print()
print("Top 1 err: ", 1 - float(correct_1) / len(cifar100_test_loader.dataset))
print("Top 5 err: ", 1 - float(correct_5) / len(cifar100_test_loader.dataset))
print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))