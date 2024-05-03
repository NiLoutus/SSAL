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
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Subset, ConcatDataset



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


# In[2]:


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# In[3]:


set_seed(seed)


# In[4]:


# Load the dataset
dataset, info = tfds.load('colorectal_histology', with_info=True, as_supervised=True)
dataset = dataset['train'].batch(len(dataset['train']))


# In[5]:


# Transform the dataset into pytorch
for images, labels in dataset:
    images_tensor = torch.tensor(images.numpy(), dtype=torch.float)
    images_tensor = images_tensor.permute(0, 3, 1, 2)
    labels_tensor = torch.tensor(labels.numpy(), dtype=torch.long)


# In[6]:


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


# In[7]:


def stratified_split(dataset, test_size=0.2):

    labels = np.array([label for _, label in dataset])

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

transform = transforms.Compose([

    transforms.ToPILImage(),  # Convert numpy array to PIL Image to apply transforms
    transforms.RandomHorizontalFlip(p=0.5),  # Apply horizontal flip with 50% probability
    transforms.RandomVerticalFlip(p=0.5),    # Apply vertical flip with 50% probability
    transforms.ToTensor(),  # Convert PIL Image back to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# In[8]:


dataset = ColorectalHistDataset(images_tensor, labels_tensor, transform)
train_subset, test_subset = stratified_split(dataset)


# In[9]:


# The base learner

class CRCClassifier(nn.Module):
    def __init__(self, num_classes=8):
        super(CRCClassifier, self).__init__()
        self.num_classes = num_classes
        resnet18 = models.resnet18(pretrained=True)
        resnet18.fc =  nn.Sequential(nn.Dropout(0.25), nn.Linear(resnet18.fc.in_features, num_classes))
        self.conv_layers = resnet18

    def forward(self, x):
        return self.conv_layers(x)


# In[10]:


def random_split(dataset, test_size=0.8):
    num_samples = len(dataset)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    split = int(np.floor(test_size * num_samples))
    train_indices, test_indices = indices[split:], indices[:split]
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)
    return train_subset, test_subset


# In[15]:


model = CRCClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
num_epochs = 100
batch_size = 40
val_loader = DataLoader(test_subset, batch_size=40, shuffle=False)


# In[16]:


def train_multistep(model, train_loader, optimizer, num_epochs):
    best_acc = 0
    for epoch in range(num_epochs):
        model.train()
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
        # print(f"val_acc:{val_acc}")
        
        if val_acc > best_acc:
            best_acc = val_acc
        
    return best_acc
        


# In[17]:


def active_sampling(model, unlabeled_subset, strategy = 'uncertainty', forward_passes = 10):
    model.eval()
    with torch.no_grad():
        all_predictions = []
        unlabeled_loader = DataLoader(unlabeled_subset, batch_size=batch_size, shuffle=False, num_workers=2)
        if strategy == 'uncertainty':
            for inputs, _ in unlabeled_loader:
                mean_predictions = []
                for _ in range(forward_passes):
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    probs = torch.softmax(outputs, dim=-1)
                    mean_predictions.append(probs.unsqueeze(0))
                mean_predictions = torch.cat(mean_predictions)
                mean_predictions = torch.mean(mean_predictions, 0)
                all_predictions.append(mean_predictions)
            all_predictions = torch.cat(all_predictions)

            safe_probabilities = all_predictions.clamp(min=1e-9)
            scores = -torch.sum(safe_probabilities * torch.log(safe_probabilities), dim=1)
            _, indices = torch.topk(scores, batch_size)
        else:
            pass
    
    return indices
                

if __name__ == "__main__":

    for i in range(10, 51):
        set_seed(seed)
        model = CRCClassifier().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        curr_train_data_portion = 0.01*i
        print(f'Currently {curr_train_data_portion*100}% of the train data is being used')
        if i == 10:
            unlabeled_subset, labeled_subset = stratified_split(train_subset, test_size=curr_train_data_portion)
            labeled_loader = DataLoader(labeled_subset, batch_size=batch_size, shuffle=True, num_workers=2)
            unlabeled_loader = DataLoader(unlabeled_subset, batch_size=batch_size, shuffle=True, num_workers=2)
            acc = train_multistep(model, labeled_loader, optimizer, num_epochs)
            #torch.save(model.state_dict(), 'semi_5_percents_combined.pth')
            indices = active_sampling(model, unlabeled_subset, 'uncertainty')
            labeled_subset = ConcatDataset([labeled_subset,Subset(unlabeled_subset, indices.tolist())])
            all_indices = torch.arange(len(unlabeled_subset)).to(device)
            mask = ~torch.isin(all_indices, indices)
            indices_to_keep = all_indices[mask]
            unlabeled_subset = Subset(dataset, indices_to_keep.tolist())
        else:
            labeled_loader = DataLoader(labeled_subset, batch_size=batch_size, shuffle=True, num_workers=2)
            unlabeled_loader = DataLoader(unlabeled_subset, batch_size=batch_size, shuffle=True, num_workers=2)
            acc = train_multistep(model, labeled_loader, optimizer, num_epochs)
            indices = active_sampling(model, unlabeled_subset, 'uncertainty')
            labeled_subset = ConcatDataset([labeled_subset,Subset(unlabeled_subset, indices.tolist())])
            all_indices = torch.arange(len(unlabeled_subset)).to(device)
            mask = ~torch.isin(all_indices, indices)
            indices_to_keep = all_indices[mask]
            unlabeled_subset = Subset(dataset, indices_to_keep.tolist())
        print(f'best_acc:{acc}')
        with open('supervised_val_accs_uncertainty_10_best.txt','a+') as f:
            f.write(f'{acc}\n')
        f.close()



