# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 02:05:03 2020

@author: wwj
"""

#%%
# load
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import torch
import torchvision
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder

import torch.nn as nn
import torch.nn.functional as F

# for imbalanced dataset
from torch.utils.data.sampler import WeightedRandomSampler

#%%
'''
前置作業：(手動)
要先建立好資料夾：
train/data/bad,good,none
以及test/data/good,bad,none
'''

def processImage(nameStr):
    
    if nameStr == 'train':
        data = np.loadtxt('train.csv',dtype=np.str,delimiter=',')[1:]
        # 更改有錯的檔案名稱
        idx = np.where(data[:,0]=='AR-200139872.jpg_NCS_modified=&exif=.jpg')
        data[idx,0] = 'AR-200139872.jpg_NCS_modified=_exif=.jpg'
        
    elif nameStr == 'test':
        data = np.loadtxt('test.csv',dtype=np.str,delimiter=',')[1:]
        
    dataAmt = len(data)

    for i in range(dataAmt):
        left = int(data[:,-4][i]) # xmin
        bottom = int(data[:,-3][i]) # ymin
        right = int(data[:,-2][i]) # xmax
        top = int(data[:,-1][i]) # ymax

        label = data[:,3][i]

        img = Image.open('../images/' + data[:,0][i])
        img_crop = img.crop((left, bottom, right, top)) 
        img_crop = img_crop.convert("RGB")
        img_crop.save(nameStr+'/data/'+label+'/'+str(i)+'.jpg',"JPEG")

processImage("train")
processImage("test")
        
#%%
transform = transforms.Compose(
    [
        transforms.Resize((32,32)), # 使圖片同樣大小
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # 將圖片轉成tensor
        transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5]) # normalization
    ]
)

#%%

trainSet = ImageFolder('train/data/',transform=transform)
print("train length:",len(trainSet),", class index:", trainSet.class_to_idx)
testSet = ImageFolder('test/data/',transform=transform)
print("test length:",len(testSet),", class index:", testSet.class_to_idx)

#%%

numClass = len(trainSet.class_to_idx) # 類別種數
className = ['bad','good','none']


#%%

trainloader = torch.utils.data.DataLoader(trainSet, 
                                          batch_size=16, 
                                          shuffle=True)
testloader = torch.utils.data.DataLoader(testSet, 
                                         batch_size=16,
                                         shuffle=False)

#%%
def imshow(img): # 顯示圖片
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
#%%
'''
網路架構
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
#         self.ada = nn.AdaptiveAvgPool2d((32,32))
        self.conv1 = nn.Conv2d(3, 5, 3, stride=1) #3:rgb #5 filters #3*3 filter size
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 10, 3, stride=1)
        self.fc1 = nn.Linear(10 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, numClass)

    def forward(self, x):
#         x = self.ada(x)
        x = self.pool(F.relu(self.conv1(x))) # 15 * 15
        x = self.pool(F.relu(self.conv2(x))) #  6 * 6
        x = x.view(-1, 10 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#%%
'''
參數設定與初始網路
'''
net = Net()

# loss 
criterion = nn.CrossEntropyLoss()

# optimizer & set learning rate
learning_rate = 0.001
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
# optimizer = optim.Adam(net.parameters(), lr=learning_rate)

epochs = 10
epoch_loss = []
epoch_train_acc = []
epoch_test_acc = []

print('Epochs: ', epochs, ', learning rate: ', learning_rate)
print(net)

#%%
'''
模型訓練
'''
for epoch in range(epochs):
    print(epoch)
    losses = 0.0
    train_acc = 0 
    test_acc = 0
    test_prediction = []
    
    # 各類別分對了的數量累計
    trainAccEachClass = [0]*numClass
    testAccEachClass = [0]*numClass
    
    # 各類別數量(從原始data之label統計)
    trainClassCount = [0]*numClass
    testClassCount = [0]*numClass
    
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        true_ans = labels.detach().numpy()
        
        optimizer.zero_grad()
        outputs = net(inputs)
        
        # training accuracy
        pred = np.argmax(outputs.detach().numpy(),axis=1)
        
        for k in range(len(pred)):
            # class count ++
            trainClassCount[true_ans[k]] += 1
            # accurate
            if true_ans[k] == pred[k]:
                train_acc += 1
                trainAccEachClass[true_ans[k]] += 1
        
        # training loss
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        losses += loss.item()
        
    # testing
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        outputs = net(inputs)
        
        true_ans = labels.detach().numpy()
        
        # testing accuracy
        pred = np.argmax(outputs.detach().numpy(),axis=1)
        
        for k in range(len(pred)):
            # class count ++
            testClassCount[true_ans[k]] += 1
            # accurate
            if true_ans[k] == pred[k]:
                test_acc += 1
                testAccEachClass[true_ans[k]] += 1
    
    epoch_loss.append(losses/len(trainSet))
    epoch_train_acc.append(train_acc/len(trainSet))
    epoch_test_acc.append(test_acc/len(testSet))
    
    print('*****************************************************')
    print('Epoch: %d/%d'%(epoch,epochs))
    print('Loss:  %.3f' % (losses/len(trainSet)) )
    print('train acc: %.3f' % (train_acc/len(trainSet)),' , ',train_acc, '/', len(trainSet))
    print('test acc: %.3f' % (test_acc/len(testSet)),' , ',test_acc, '/', len(testSet) )
    print('train acc each class: ',(np.array(trainAccEachClass) / np.array(trainClassCount)))
    print('test acc each class: ',(np.array(testAccEachClass) / np.array(testClassCount)))
    print('*****************************************************')

print("Training finished!!!")

#%%
'''
結果分析
'''
# 訓練結果
print('Loss:  %.3f' % (losses/len(trainSet)) )
print('train acc: %.3f' % (train_acc/len(trainSet)),' , ',train_acc, '/', len(trainSet))
print('test acc: %.3f' % (test_acc/len(testSet)),' , ',test_acc, '/', len(testSet) )
print('train acc each class: ',(np.array(trainAccEachClass) / np.array(trainClassCount)))
print('test acc each class: ',(np.array(testAccEachClass) / np.array(testClassCount)))


# plot the loss, and training & testing acc
plt.style.use("default")
plt.figure(figsize=(6,4))
plt.title("Accuracy")

plt.xlim(-1,10)
plt.ylim(0,1)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.plot(epoch_train_acc, label='train')
plt.plot(epoch_test_acc, label='test')
plt.legend(['train','test'])
plt.show()

plt.figure(figsize=(6,4))
plt.title("Learning Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(epoch_loss, color='g')
plt.show()

#%%

'''
圖片標記框框
'''
testData = np.loadtxt('test.csv',dtype=np.str,delimiter=',')[1:]

labels_name = ['bad','good','none']
labelcolors = ['r','g','b']

img_list = np.unique(testData[:,0])
for image in range(len(img_list)):
    img_name = img_list[image]
    print(img_name)
    idx = np.where(testData[:,0] == img_name)
    showData = testData[idx,:]
    
    predict_labels = np.array(np.array(test_prediction)[idx]) # predictions
    true_labels = showData[0][:,3]
    
    left = np.array(list(map(int, showData[0][:,4]))) # xmin
    bottom = np.array(list(map(int, showData[0][:,5]))) # ymin
    right = np.array(list(map(int, showData[0][:,6])))# xmax
    top = np.array(list(map(int, showData[0][:,7]))) # ymax

    width = np.array(right)-np.array(left)
    height = np.array(top)-np.array(bottom)
    
    # show image
    img = np.array(Image.open('../images/'+img_name), dtype=np.uint8)

    fig,ax = plt.subplots(1,figsize=(10,6))
    ax.imshow(img)

    # rectangle
    for i in range(len(showData[0])):

        rect = patches.Rectangle((left[i],top[i]-60),
                                 width[i],
                                 height[i],
                                 linewidth=2,
                                 edgecolor=labelcolors[int(predict_labels[i])],
                                 facecolor='none',
                                 linestyle='-') #--
        ax.add_patch(rect)

        ax.text(left[i],top[i]-60, 'pred:'+labels_name[predict_labels[i]]+'\ntrue:'+true_labels[i],
                color=labelcolors[int(predict_labels[i])],
                bbox=dict(facecolor=(1,1,1,0.9), edgecolor=labelcolors[int(predict_labels[i])]),
                fontsize=8)

    plt.show()

#%%
'''
處理imbalanced dataset
'''
# 各類別的數量(由剛才的訓練過程統計)
print(trainClassCount, testClassCount)

#%%
'''
設定sample比重種類：三種類別、三種weights
'''

# train
classWeights = [1]*3
classWeights = np.array(classWeights) / np.array(trainClassCount)
print(classWeights)
trainSampleWeights = []
for i in range(len(trainSet)):
    label = trainSet[i][1]
    trainSampleWeights.append(classWeights[label])
print(trainSampleWeights)

# test
classWeights = [1]*3
classWeights = np.array(classWeights) / np.array(testClassCount)
print(classWeights)
testSampleWeights = []
for i in range(len(testSet)):
    label = testSet[i][1]
    testSampleWeights.append(classWeights[label])
print(testSampleWeights)


#%%
'''
WeightedRandomSampler
'''
trainsampler = WeightedRandomSampler(trainSampleWeights, 
                                num_samples =len(trainSet), 
                                replacement = True)
testsampler = WeightedRandomSampler(testSampleWeights, 
                                num_samples =len(testSet), 
                                replacement = True)

#%%
trainloader = torch.utils.data.DataLoader(trainSet, 
                                          batch_size=16, 
                                         sampler = trainsampler)
testloader = torch.utils.data.DataLoader(testSet, 
                                         batch_size=16,
                                        sampler = testsampler)

#%%
net = Net()

# loss 
criterion = nn.CrossEntropyLoss()

# optimizer & set learning rate
learning_rate = 0.001
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
# optimizer = optim.Adam(net.parameters(), lr=learning_rate)

epochs = 10
epoch_loss = []
epoch_train_acc = []
epoch_test_acc = []

print('Epochs: ', epochs, ', learning rate: ', learning_rate)
print(net)

#%%
for epoch in range(epochs):
    print(epoch)
    losses = 0.0
    train_acc = 0 
    test_acc = 0
    test_prediction = []
    
    # 各類別分對了的數量累計
    trainAccEachClass = [0]*numClass
    testAccEachClass = [0]*numClass
    
    # 各類別數量(從原始data之label統計)
    trainClassCount = [0]*numClass
    testClassCount = [0]*numClass
    
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        true_ans = labels.detach().numpy()
        
        optimizer.zero_grad()
        outputs = net(inputs)
        
        # training accuracy
        pred = np.argmax(outputs.detach().numpy(),axis=1)
        
        for k in range(len(pred)):
            # class count ++
            trainClassCount[true_ans[k]] += 1
            # accurate
            if true_ans[k] == pred[k]:
                train_acc += 1
                trainAccEachClass[true_ans[k]] += 1
        
        # training loss
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        losses += loss.item()
        
    # testing
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        outputs = net(inputs)
        
        true_ans = labels.detach().numpy()
        
        # testing accuracy
        pred = np.argmax(outputs.detach().numpy(),axis=1)
        
        for k in range(len(pred)):
            # class count ++
            testClassCount[true_ans[k]] += 1
            # accurate
            if true_ans[k] == pred[k]:
                test_acc += 1
                testAccEachClass[true_ans[k]] += 1
    
    epoch_loss.append(losses/len(trainSet))
    epoch_train_acc.append(train_acc/len(trainSet))
    epoch_test_acc.append(test_acc/len(testSet))
    
    print('*****************************************************')
    print('Epoch: %d/%d'%(epoch,epochs))
    print('Loss:  %.3f' % (losses/len(trainSet)) )
    print('train acc: %.3f' % (train_acc/len(trainSet)),' , ',train_acc, '/', len(trainSet))
    print('test acc: %.3f' % (test_acc/len(testSet)),' , ',test_acc, '/', len(testSet) )
    print('train acc each class: ',(np.array(trainAccEachClass) / np.array(trainClassCount)))
    print('test acc each class: ',(np.array(testAccEachClass) / np.array(testClassCount)))
    print('*****************************************************')

print("Training finished!!!")

#%%
'''
結果分析
'''
# 訓練結果
print('Loss:  %.3f' % (losses/len(trainSet)) )
print('train acc: %.3f' % (train_acc/len(trainSet)),' , ',train_acc, '/', len(trainSet))
print('test acc: %.3f' % (test_acc/len(testSet)),' , ',test_acc, '/', len(testSet) )
print('train acc each class: ',(np.array(trainAccEachClass) / np.array(trainClassCount)))
print('test acc each class: ',(np.array(testAccEachClass) / np.array(testClassCount)))
