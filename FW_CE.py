import argparse
import sys
import builtins
import os
import random
import shutil
import time
import warnings
from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
from torchvision.datasets import CIFAR10
from cifar_noisy import CIFAR10_noisy
from sklearn import manifold
import numpy as np
from sklearn import manifold
from model import Model
np.random.seed(0)

parser = argparse.ArgumentParser(description='Cross Entropy')
parser.add_argument('--model_path', type=str, default='results/128_0.5_200_512_1000_model.pth',
                    help='The pretrained model path')
parser.add_argument('--batch_size', type=int, default=256, help='Number of images in each mini-batch')
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--epochs', type=int, default=150, help='Number of sweeps over the dataset to train')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.6)
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric,instance]', default='symmetric')
parser.add_argument('--simclr_pretrain', action='store_true')
parser.add_argument('--finetune_fc_only', action='store_true')
parser.add_argument('--down_sample', action='store_true')
args = parser.parse_args()




class Net(nn.Module):
    def __init__(self, num_class, pretrained_path):
        super(Net, self).__init__()

        # encoder
        self.f = Model().f
        # classifier
        self.fc = nn.Linear(512, num_class, bias=True)
        if pretrained_path is not None:
            self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out

if args.simclr_pretrain:
    model = Net(num_class=args.num_classes, pretrained_path=args.model_path).cuda()
else:
    model = Net(num_class=args.num_classes, pretrained_path=None).cuda()

if args.finetune_fc_only:
    for param in model.f.parameters():
        param.requires_grad = False



train_cifar10_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_cifar10_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])





train_dataset = CIFAR10_noisy(root='./data/',indexes = None,
                                train=True, down_sample = args.down_sample,
                                transform = train_cifar10_transform,
                                noise_type= args.noise_type,noise_rate=args.noise_rate, random_state=0)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                  batch_size = args.batch_size, 
                                  num_workers=32,
                                  shuffle=True,pin_memory=True)

test_dataset = CIFAR10(root='data', train=False, transform=test_cifar10_transform, download=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                  batch_size = args.batch_size, 
                                  num_workers=32,
                                  shuffle=False,pin_memory=True)

model.cuda()

criterion = nn.CrossEntropyLoss().cuda()


if args.simclr_pretrain:
    alpha_plan = [0.001] * args.epochs
    optimizer = optim.Adam(model.parameters(), lr=0.001)
else:
    alpha_plan = [0.01] * 40 + [0.001] * (args.epochs - 40)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

def adjust_learning_rate(optimizer, epoch,alpha_plan):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]

best_acc = [0]
def validate(val_loader, model, criterion):
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = Variable(images).cuda()
            # compute output
            logits = model(images)
            outputs = F.softmax(logits, dim=1)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred.cpu() == labels).sum()
            acc = 100*float(correct)/float(total) 
    return acc


def find_trans_mat(model):
    # estimate each component of matrix T based on training with noisy labels
    print("estimating transition matrix...")
    output_ = torch.tensor([]).float().cuda()

    # collect all the outputs
    with torch.no_grad():
        for batch_idx, (data, label,true_label,idx) in enumerate(train_loader):
            data = torch.tensor(data).float().cuda()
            outputs = model(data)
            outputs = F.softmax(outputs, dim=1)
            output_ = torch.cat([output_, outputs], dim=0)

    # find argmax_{x^i} p(y = e^i | x^i) for i in C
    hard_instance_index = output_.argmax(dim=0)
    trans_mat_ = torch.tensor([]).float()

    # T_ij = p(y = e^j | x^i) for i in C j in C
    for i in range(args.num_classes):
        trans_mat_ = torch.cat([trans_mat_, output_[hard_instance_index[i]].cpu()], dim=0)

    trans_mat_ = trans_mat_.reshape(args.num_classes, args.num_classes)

    return trans_mat_

def forward_loss(output, target, trans_mat):
    # l_{forward}(y, h(x)) = l_{ce}(y, h(x) @ T)
    outputs = F.softmax(output, dim=1)
    outputs = outputs @ trans_mat.cuda()
    outputs = torch.log(outputs)
    loss = criterion(outputs, target)
    return loss



model.load_state_dict(torch.load('ce_'+args.noise_type+'_'+str(args.noise_rate)+'.pth')['state_dict'])
trans_mat = find_trans_mat(model)

for epoch in range(args.epochs):
    if args.finetune_fc_only:
        model.eval()
    else: 
        model.train()
    adjust_learning_rate(optimizer, epoch, alpha_plan)
    for i, (images, labels,true_labels,indexes) in enumerate(train_loader):
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        output = model(images)
        loss = forward_loss(output, labels,trans_mat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    acc1 = validate(test_loader, model, criterion)
    if acc1>best_acc[0]:
        best_acc[0] = acc1
    print('current epoch',epoch)
    print('best acc',best_acc[0])
    print('last acc', acc1)



