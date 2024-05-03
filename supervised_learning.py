#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import Subset, SubsetRandomSampler



CUDA = True
CUDA = CUDA and torch.cuda.is_available()
seed = 42
print("PyTorch version: {}".format(torch.__version__))
if CUDA:
    print("CUDA version: {}\n".format(torch.version.cuda))

if CUDA:
    torch.cuda.manual_seed(seed)
device = torch.device("cuda:0" if CUDA else "cpu")
cudnn.benchmark = True


# In[21]:


while True:
    try:
        torch.randn(10).cuda()
        break
    except:
        pass


# In[2]:


torch.randn(10).cuda()


# In[6]:


# Load the dataset
dataset, info = tfds.load('colorectal_histology', with_info=True, as_supervised=True)
dataset = dataset['train'].batch(len(dataset['train']))


# In[7]:


# Transform the dataset into pytorch
for images, labels in dataset:
    images_tensor = torch.tensor(images.numpy(), dtype=torch.float)
    images_tensor = images_tensor.permute(0, 3, 1, 2)
    labels_tensor = torch.tensor(labels.numpy(), dtype=torch.long)


# In[8]:


class ColorectalHistDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx].clone().detach()
        label = self.labels[idx].clone().detach()

        if self.transform:
            image = self.transform(image)

        return image, label


# In[9]:


def stratified_split(dataset, test_size=0.2):

    labels = np.array([ins[-1] for ins in dataset])

    # Indices for each class
    class_indices = [np.where(labels == class_label)[0] for class_label in np.unique(labels)]

    # Split each class's indices into train and test
    train_indices, test_indices = [], []
    for indices in class_indices:
        np.random.shuffle(indices)
        split = int(np.floor(test_size * len(indices)))
        train_indices.extend(indices[split:])
        test_indices.extend(indices[:split])

    # Create subset for train and test
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    return train_subset, test_subset


# In[10]:


# Transformation for supervised learning
transform = transforms.Compose([

    transforms.ToPILImage(),  # Convert numpy array to PIL Image to apply transforms
    transforms.RandomHorizontalFlip(p=0.5),  # Apply horizontal flip with 50% probability
    transforms.RandomVerticalFlip(p=0.5),    # Apply vertical flip with 50% probability
    transforms.RandomResizedCrop(size=(150, 150), antialias=True),
    transforms.ToTensor(),  # Convert PIL Image back to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = ColorectalHistDataset(images_tensor, labels_tensor, transform)
train_subset, test_subset = stratified_split(dataset)


# In[11]:


# The base learner

class CRCClassifier(nn.Module):
    def __init__(self, num_classes=8):
        super(CRCClassifier, self).__init__()
        self.num_classes = num_classes
        resnet18 = models.resnet18(weights=True)
        resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)
        self.conv_layers = resnet18

    def forward(self, x):
        return self.conv_layers(x)

model = CRCClassifier().to(device)


# In[ ]:


criterion = nn.CrossEntropyLoss()
num_epochs = 100
curr_percentage = 0.01
max_epoch_accs = []
max_val_accs = []
min_val_losses = []
min_epoch_losses = []

for i in range(41,51):#Loop to run 50 times - 1 for 1% of the train dataset
    curr_train_data_len = int(len(train_subset)*curr_percentage*i)
    print(f'Currently {curr_percentage*i*100}% of the train data is being used')
    indices = torch.randperm(len(train_subset))[:curr_train_data_len]
    curr_train_data = Subset(train_subset, indices)
    train_loader = DataLoader(curr_train_data, batch_size=40, shuffle=True, num_workers=2)
    val_loader = DataLoader(test_subset, batch_size=40, shuffle=False)
    model = CRCClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    epoch_accs = []
    epoch_losses = []
    validation_accs = []
    validation_losses = []
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(curr_train_data)
        epoch_acc = running_corrects.double() / len(curr_train_data)
        epoch_accs.append(epoch_acc.item())
        epoch_losses.append(epoch_loss)
        
      # Validation phase
        model.eval()  # Set model to evaluate mode
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                
    val_loss = val_loss / len(test_subset)
    val_acc = val_corrects.double() / len(test_subset)
    validation_losses.append(val_loss)
    validation_accs.append(val_acc.item())
    max_epoch_idx = np.argmax(epoch_accs)
    max_val_idx = np.argmax(validation_accs)
    print(f' Epoch Loss: {epoch_losses[max_epoch_idx]}, Max Epoch Acc: {epoch_accs[max_epoch_idx]}')
    print(f'Max Validation Loss: {validation_losses[max_val_idx]}, Max Validation Acc: {validation_accs[max_val_idx]}')
    max_epoch_accs.append(epoch_accs[max_epoch_idx])
    max_val_accs.append(validation_accs[max_val_idx])
    min_epoch_losses.append(epoch_losses[max_epoch_idx])
    min_val_losses.append(validation_losses[max_val_idx])
    with open('sup_val_losses.txt','a+') as f:
        for val in min_val_losses:
            f.write(f'{val}\n')
    f.close()

    with open('sup_epoch_losses.txt','a+') as f:
        for val in min_epoch_losses:
            f.write(f'{val}\n')
    f.close()

    with open('sup_val_accs.txt','a+') as f:
        for val in max_val_accs:
            f.write(f'{val}\n')
    f.close()

    with open('sup_epoch_accs.txt','a+') as f:
        for val in max_epoch_accs:
            f.write(f'{val}\n')
    f.close()


# In[15]:


with open('sup_val_losses.txt','w') as f:
    for val in min_val_losses:
        f.write(f'{val}\n')
f.close()

with open('sup_epoch_losses.txt','w') as f:
    for val in min_epoch_losses:
        f.write(f'{val}\n')
f.close()

with open('sup_val_accs.txt','w') as f:
    for val in max_val_accs:
        f.write(f'{val}\n')
f.close()

with open('sup_epoch_accs.txt','w') as f:
    for val in max_epoch_accs:
        f.write(f'{val}\n')
f.close()


# In[ ]:




