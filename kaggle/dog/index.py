import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
import pickle
from skimage import io, transform
from PIL import Image
from dataset import DogDataset
import csv

data_transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
use_gpu = torch.cuda.is_available()

def train(epoch_num=2):
    datasets = {x: DogDataset('labels/'+x+'.csv', 'datasets/dog-breed-identification/train/', data_transform[x]) for x in ['train', 'val']}
    data_loaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 120)
    if use_gpu:
        model_ft = model_ft.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataset_sizes, data_loaders, num_epochs=epoch_num, resume='checkpoint.pth.tar')
    output=open('data.pkl', 'wb')
    pickle.dump(model_ft, output)

def train_model(model, criterion, optimizer, scheduler, dataset_sizes, data_loaders, num_epochs=25, resume=''):
    resume_epoch=0
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            resume_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs-resume_epoch):
        print('Epoch {}/{}'.format(epoch+resume_epoch, num_epochs-1))
        print('-'*10)
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)
            running_loss = 0.0
            running_corrects = 0

            batch_num = 1
            for data in data_loaders[phase]:
                inputs = data[0]
                labels = data[1]
                if(batch_num%100 ==0):
                    print("Batch#"+str(batch_num))
                batch_num+=1
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += loss.data.item()*inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = torch.true_divide(running_loss, dataset_sizes[phase])
            epoch_acc = torch.true_divide(running_corrects, dataset_sizes[phase])
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            save_checkpoint({
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            })
        print()
    time_ekapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_ekapsed // 60, time_ekapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def test(resume_file):
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 120)

    best_wts = torch.load(resume_file)
    model.load_state_dict(best_wts['state_dict'])
    if use_gpu:
        model = model.cuda()
    dataset = DogDataset(csv_file='labels/test.csv', root_dir='datasets/dog-breed-identification/test/', transform=data_transform['val'], mode='test')

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    test_model(model, data_loader, len(dataset))

def test_model(model, data_loader, data_size):
    model.train(False)
    with open('predictions.csv', 'w') as prediction_file:
        csvwriter = csv.writer(prediction_file)
        num_preds=0
        for data in data_loader:
            if num_preds % 100 == 0:
                print('Predictions: {}/{}'.format(num_preds, len(data_loader)-1))
                print('-'*10)
                print('')
            inputs = data[0]
            ids = data[1]
            if use_gpu:
                inputs = Variable(inputs.cuda())
            else:
                inputs = Variable(inputs)
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            for i in range(len(ids)):
                id = ids[i]
                pred = np.zeros(120)
                pred[preds[i]] = 1
                row = [id]+pred.tolist()
                csvwriter.writerow(row)
            num_preds+=1

def visualize_model(model, dataloader, class_names, num_images=6):
    images_so_far = 0
    for data in dataloader:
        inputs, ids = data
        if use_gpu:
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        for j in range(inputs.size()[0]):
            images_so_far += 1
            img_name = os.path.join('datasets/dog-breed-identification/test/', ids[j] + '.jpg')
            img = Image.open(img_name)
            plt.imshow(img)
            plt.title('predicted: {}'.format(class_names[preds[j]+1]))
            print('wrote prediction#' + str(images_so_far))
            plt.savefig('predictions/prediction#' + str(images_so_far) + '.jpg')
            if images_so_far == num_images:
                return

def visualize(resume_file):
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 120)

    best_wts = torch.load(resume_file)
    model.load_state_dict(best_wts['state_dict'])
    if use_gpu:
        model = model.cuda()
    dataset = DogDataset('labels/test.csv', 'datasets/dog-breed-identification/test/', data_transform['val'], mode='test')
    class_name = pd.read_csv('labels/class_names.csv', header=None)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    visualize_model(model, data_loader, class_name.values[0], num_images=10)
train()
# test('checkpoint.pth.tar')
# visualize('checkpoint.pth.tar')