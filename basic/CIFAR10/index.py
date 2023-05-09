import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

train_batch_size = 4
test_batch_size = 4
num_workers = 0
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
lr = 0.001
momentum = 0.9
epochs = 10

mean = [0.485, 0.456, 0.406]
std= [0.229, 0.224, 0.225]
custom_transform = transforms.transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

train_data = torchvision.datasets.CIFAR10('./data', train=True, transform=custom_transform, target_transform=None, download=True)
test_data = torchvision.datasets.CIFAR10('./data', train=False, transform=custom_transform, target_transform=None, download=True)
train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def displayImage():
    plt.figure()
    def imshow(img):
        npimg = img.numpy()
        for i in range(len(mean)):
            npimg[i] = npimg[i] * std[i] + mean[i]
        plt.imshow(np.transpose(npimg, [1,2,0]))
        plt.show()

    examples = enumerate(train_loader)
    idx, (examples_data, examples_target) = next(examples)
    imshow(torchvision.utils.make_grid(examples_data))
    print('-'*20, 'expamles','-'*20)
    print('examples_target.shape:{}'.format(examples_target.shape))
    print('examples_target[0]:{}'.format(examples_target[0]))
    print('examples_data.shape:{}'.format(examples_data.shape))

class CNNet(nn.Module):
    def __init__(self):
        super(CNNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=36, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(1296, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x))) # 4*3*32*32 -> 4*16*28*28 -> 4*16*14*14
        x = self.pool2(F.relu(self.conv2(x))) # 4*16*14*14 -> 4*36*12*12 -> 4*36*6*6
        x = x.view(-1, 36*6*6)
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return x

model = CNNet()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

losses = []
acces = []
resume_epoch = 0
if os.path.isfile('checkpoint.pth.tar'):
    print("=> loading checkpoint '{}'".format('checkpoint.pth.tar'))
    checkpoint = torch.load('checkpoint.pth.tar')
    resume_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {})".format('checkpoint.pth.tar', checkpoint['epoch']))

for epoch in range(resume_epoch, epochs):
    train_loss = 0
    num_correct = 0
    model.train()
    for i, data in enumerate(train_loader):
        img, label = data
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, pred = out.max(1)
        num_correct += (pred == label).sum()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f ' % (epoch + 1, i+1, train_loss/(i+1)))
    acces.append(torch.true_divide(num_correct,len(train_loader)*train_batch_size))
    losses.append(torch.true_divide(train_loss, len(train_loader) * train_batch_size))
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, 'checkpoint.pth.tar')
plt.title('Train Acc')
plt.plot(np.arange(len(acces)), acces)
plt.legend(['Train Acc'], loc='upper right')
plt.show()

eval_loss = 0
eval_acc = 0
class_correct = list(0. for i in range (len(classes)))
class_total = list(0. for i in range(len(classes)))
total = 0
num_correct = 0
model.eval()
with torch.no_grad():
    for img, label in test_loader:
        img, label = img.to(device), label.to(device)
        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.item()
        _, pred = out.max(1)
        num_correct += (pred == label).sum()
        c = (pred == label).squeeze()
        total += label.size(0)

        for i in range(test_batch_size):
            class_correct[label[i]] += c[i].item()
            class_total[label[i]] += 1
    print('total:{}'.format(total))
    print('len(test_loader):{}'.format(len(test_loader)))
    eval_acc = torch.true_divide(num_correct, total)
    for i in range(len(classes)):
        print('accuracy of {}:{}%'.format(classes[i], 100*class_correct[i]/class_total[i]))
    print('-'*20)
    print('epoch:{}, eval_loss:{:.4f}, eval_acc:{:.4f}'.format(epochs, eval_loss/total, eval_acc))
    print('Accuracy of the network on the %d test images: %d %%' % (total, 100*eval_acc))