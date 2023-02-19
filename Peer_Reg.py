import sys
import argparse
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
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import CIFAR10
from cifar_noisy import CIFAR10_noisy
from sklearn import manifold
from model import Model
import numpy as np
from loss import RkdDistance, RKdAngle,Info_NCE

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
parser.add_argument('--reg', type = str, default='rkd_dis')
parser.add_argument('--simclr_pretrain', action='store_true')
args = parser.parse_args()

def f_beta(epoch):
    beta1 = np.linspace(0.0, 0.0, num=10)
    beta2 = np.linspace(0.0, 2, num=30)
    beta3 = np.linspace(2, 2, num=160)
 
    beta = np.concatenate((beta1,beta2,beta3),axis=0) 
    return beta[epoch]




class Net(nn.Module):
    def __init__(self, num_class, pretrained_path):
        super(Net, self).__init__()

        # encoder
        self.f = Model().f
        # classifier
        self.g = Model().g
        self.fc = nn.Linear(512, num_class, bias=True)
        if pretrained_path is not None:
            self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        g_out = self.g(feature)
        out = self.fc(feature)
        return g_out, out

if args.simclr_pretrain:
    model = Net(num_class=args.num_classes, pretrained_path=args.model_path).cuda()
else:
    model = Net(num_class=args.num_classes, pretrained_path=None).cuda()


train_cifar10_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_cifar10_strong_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_cifar10_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])



class CIFAR10_noisy_3img(CIFAR10_noisy):

    def __init__(self,root,indexes = None,
                                train=True, 
                                transform = None,
                                strong_transform = None,
                                noise_type='symmetric',noise_rate=0.6, random_state=0):
        super(CIFAR10_noisy_3img, self).__init__(root, indexes=indexes, train=train,
                 transform=transform,noise_type=noise_type,noise_rate = noise_rate,random_state=random_state)
        
        self.strong_transform = strong_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        label = self.train_noisy_labels[index]
        true_label = self.true_labels[index]
        img = self.train_data[index]
        img = Image.fromarray(img)
        img1 = self.transform(img)
        img2 = self.strong_transform(img)
        img3 = self.strong_transform(img)
        return img1,img2,img3,label,true_label,index




train_dataset = CIFAR10_noisy_3img(root='./data/',indexes = None,
                                train=True, 
                                transform = train_cifar10_transform,
                                strong_transform = train_cifar10_strong_transform,
                                noise_type=args.noise_type,noise_rate=args.noise_rate, random_state=0)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                  batch_size = args.batch_size, 
                                  num_workers=32,drop_last=True,
                                  shuffle=True,pin_memory=True)


peer_dataset = CIFAR10_noisy(root='./data/',indexes = None,
                                train=True,
                                transform = train_cifar10_transform,
                                noise_type= args.noise_type,noise_rate=args.noise_rate, random_state=0)



peer_loader = torch.utils.data.DataLoader(dataset=peer_dataset,
                                  batch_size = args.batch_size, 
                                  num_workers=32,
                                  shuffle=True,drop_last = True,pin_memory=True)


test_dataset = CIFAR10(root='data', train=False, transform=test_cifar10_transform, download=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                  batch_size = args.batch_size, 
                                  num_workers=32,
                                  shuffle=False,pin_memory=True)


model.cuda()

class CrossEntropyLossStable(nn.Module):
    def __init__(self, reduction='mean', eps=1e-5):
        super(CrossEntropyLossStable, self).__init__()
        self._name = "Stable Cross Entropy Loss"
        self._eps = eps
        self._softmax = nn.Softmax(dim=-1)
        self._nllloss = nn.NLLLoss(reduction=reduction)

    def forward(self, outputs, labels):
        return self._nllloss( torch.log( self._softmax(outputs) + self._eps ), labels )




reg_factory = {'rkd_dis':RkdDistance(),'rkd_angle':RKdAngle()}
base_loss = CrossEntropyLossStable().cuda()
self_criterion = Info_NCE() 
reg_criterion = reg_factory[args.reg]





if args.simclr_pretrain:
    alpha_plan = [0.001] * args.epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
else:
    alpha_plan = [0.1] * 50 + [0.01] * (args.epochs - 50)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

def adjust_learning_rate(optimizer, epoch,alpha_plan):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]

best_acc = [0]
def validate(val_loader, model):
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = Variable(images).cuda()
            # compute output
            features, logits = model(images)
            outputs = F.softmax(logits, dim=1)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred.cpu() == labels).sum()
            acc = 100*float(correct)/float(total) 

    return acc


model.load_state_dict(torch.load('ce_reg_'+args.noise_type+'_'+str(args.noise_rate)+'.pth')['state_dict'])

for epoch in range(args.epochs):
    model.train()
    adjust_learning_rate(optimizer, epoch, alpha_plan)
    peer_iter = iter(peer_loader)
    for i, (images1, images2,images3,labels,true_labels,indexes) in enumerate(train_loader):
        images1 = Variable(images1).cuda()
        images2 = Variable(images2).cuda()
        images3 = Variable(images3).cuda()
        labels = Variable(labels).cuda()
        features1, output1 = model(images1)
        features2, output2 = model(images2)
        features3, output3 = model(images3)
        try:
            peer_images = peer_iter.next()[0]
            peer_labels = peer_iter.next()[1]
        except StopIteration:
            peer_iter = iter(peer_loader)
            peer_images = peer_iter.next()[0]
            peer_labels = peer_iter.next()[1]
        peer_images = Variable(peer_images).cuda()
        peer_labels = Variable(peer_labels).cuda()
        peer_feature, peer_output = model(peer_images)

        #loss = base_loss(output1, labels) - f_beta(epoch)*base_loss(peer_output,peer_labels) + self_criterion(features2,features3) + reg_criterion(output1,features1)
        loss = base_loss(output1, labels) - base_loss(peer_output,peer_labels)  + self_criterion(features2,features3) + reg_criterion(output1,features1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    acc1 = validate(test_loader, model)
    if acc1 > best_acc[0]:
        best_acc[0] = acc1
    print('current epoch',epoch)
    print('best acc',best_acc[0])
    print('last acc', acc1)









