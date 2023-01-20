# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 01:55:13 2022

@author: shenyutao
"""

from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint,TensorBoard,LambdaCallback
from keras.layers import Input,Dropout, Dense,GlobalAveragePooling2D
from keras.models import Sequential,Model
from keras.applications.resnet import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from tqdm import tqdm
import seaborn as sns
import numpy as np
import itertools 
import datetime
import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import copy

import cv2
import os
import io

labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

x_train = [] # training images.
y_train  = [] # training labels.
x_test = [] # testing images.
y_test = [] # testing labels.

image_size = 200


for label in labels:
    trainPath = os.path.join('./cleaned/Training',label)
    for file in tqdm(os.listdir(trainPath)):
        image = cv2.imread(os.path.join(trainPath, file),0) # load images in gray.
        image = cv2.bilateralFilter(image, 2, 50, 50) # remove images noise.
        image = cv2.applyColorMap(image, cv2.COLORMAP_BONE) # produce a pseudocolored image.
        image = cv2.resize(image, (image_size, image_size)) # resize images into 150*150.
        x_train.append(image)
        y_train.append(labels.index(label))
    
    testPath = os.path.join('./cleaned/Testing',label)
    for file in tqdm(os.listdir(testPath)):
        image = cv2.imread(os.path.join(testPath, file),0)
        image = cv2.bilateralFilter(image, 2, 50, 50)
        image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
        image = cv2.resize(image, (image_size, image_size))
        x_test.append(image)
        y_test.append(labels.index(label))


x_train = np.array(x_train) / 255.0 # normalize Images into range 0 to 1.
x_test = np.array(x_test) / 255.0


x_train = x_train[0:1400]
y_train = y_train[0:1400]
x_train, y_train = shuffle(x_train,y_train, random_state=42) 
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

class Dataset:
    def __init__(self, data, target, polluted=False, rho=0.0):
        self.data = data.float() / torch.max(data)
        print(list(target.shape))
        if not polluted:
            self.clean_target = target
            self.dirty_target = None
            self.clean = np.ones(list(target.shape)[0])
        else:
            self.clean_target = None
            self.dirty_target = target
            self.clean = np.zeros(list(target.shape)[0])
        self.polluted = polluted
        self.rho = rho
        self.set = set(target.numpy().tolist())

    def data_polluting(self, rho):
        assert self.polluted == False and self.dirty_target is None
        number = self.data.shape[0]
        number_list = list(range(number))
        random.shuffle(number_list)
        self.dirty_target = copy.deepcopy(self.clean_target)
        for i in number_list[:int(rho * number)]:
            dirty_set = copy.deepcopy(self.set)
            dirty_set.remove(int(self.clean_target[i]))
            self.dirty_target[i] = random.randint(0, len(dirty_set))
            self.clean[i] = 0
        self.polluted = True
        self.rho = rho
        
x_train_tensor = torch.from_numpy(x_train)
y_train_tensor = torch.Tensor(y_train)
x_val_tensor = torch.from_numpy(x_val)
y_val_tensor = torch.Tensor(y_val)
x_test_tensor = torch.from_numpy(x_test)
y_test_tensor = torch.Tensor(y_test)
tr = Dataset(x_train_tensor, y_train_tensor)
val = Dataset(x_val_tensor, y_val_tensor)
test = Dataset(x_test_tensor, y_test_tensor)
tr.data_polluting(0.5) #pollute here
tr.data = tr.data.permute(0, 3, 1, 2)
val.data = val.data.permute(0, 3, 1, 2)
test.data = test.data.permute(0, 3, 1, 2)

import torchvision.models as models
model = models.resnet18(pretrained=True)

model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])

for param in model.parameters():
    param.requires_grad = True

# Parameters of newly constructed modules have requires_grad=True by default
in_features = model.module.fc.in_features
model.module.fc = nn.Linear(in_features, 4)

device = ("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

from torch.utils.data import DataLoader
test_data = []
for i in range(len(test.data)):
   test_data.append([test.data[i], test.clean_target[i]])
test_loader = DataLoader(dataset=test_data, shuffle=True, batch_size=4, num_workers=4, pin_memory=True)
val_data = []
for i in range(len(val.data)):
   val_data.append([val.data[i], val.clean_target[i]])
val_loader = DataLoader(dataset=val_data, shuffle=True, batch_size=4, num_workers=4, pin_memory=True)
tr_data = []
for i in range(len(tr.data)):
   tr_data.append([tr.data[i], tr.dirty_target[i]])
train_loader = DataLoader(dataset=tr_data, shuffle=True, batch_size=4, num_workers=4, pin_memory=True)

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        images, labels = data

        images = images.to(device)
        labels = labels.to(device)

        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Before training, Accuracy of test images: %d %%' % (
    100 * correct / total))
print("Correct: ", correct)
print("Total: ", total)


class Net_x(torch.nn.Module):
    def __init__(self, tr):
        super(Net_x, self).__init__()
        self.x = torch.nn.Parameter(torch.zeros(4).to(device).requires_grad_(True))

    def forward(self, y):
        # if torch.norm(torch.sigmoid(self.x), 1) > 2500:
        #     y = torch.sigmoid(self.x) / torch.norm(torch.sigmoid(self.x), 1) * 2500 * y
        # else:
        y = torch.sigmoid(self.x) * y
        y = y.mean()
        return y

x = Net_x(tr)
x_opt = torch.optim.Adam(x.parameters(), lr=0.1)

import torch.optim as optim
criterion = nn.CrossEntropyLoss()
y_opt = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
y = model


for x_itr in range(20):
    x_opt.zero_grad()
    print(x_itr)
    for epoch in range(1):  # loop over the dataset multiple times
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
    
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            # zero the parameter gradients
            y_opt.zero_grad()
    
            # forward + backward + optimize
            outputs = y(inputs)
            loss = x(F.cross_entropy(outputs, labels.long(), reduction='none'))
            loss.backward()
            y_opt.step()
    
            
    for data, labels in val_loader:
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()
        
        target = y(data)
        loss = criterion(target, labels.long())
        loss.backward()
    x_opt.step()
        
        
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data

            images = images.to(device)
            labels = labels.to(device)

            # calculate outputs by running images through the network
            outputs = y(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('After ', x_itr + 1, 'iterations, Accuracy of test images: %d %%' % (
        100 * correct / total))
    print("Correct: ", correct)
    print("Total: ", total)    
        
print('Finished Training')

