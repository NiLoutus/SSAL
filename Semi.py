#!/usr/bin/env python
# coding: utf-8

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
import torch.nn.functional as F
from torch.utils.data import Subset, ConcatDataset
from randaugment import RandAugmentMC

CUDA = CUDA and torch.cuda.is_available()
seed = 42
print("PyTorch version: {}".format(torch.__version__))
if CUDA:
    print("CUDA version: {}\n".format(torch.version.cuda))

if CUDA:
    torch.cuda.manual_seed(seed)
device = torch.device("cuda:0" if CUDA else "cpu")
cudnn.benchmark = True

# Load the dataset
dataset, info = tfds.load('colorectal_histology', with_info=True, as_supervised=True)
dataset = dataset['train'].batch(len(dataset['train']))

# Transform the dataset into pytorch
for images, labels in dataset:
    images_tensor = torch.tensor(images.numpy(), dtype=torch.float)
    images_tensor = images_tensor.permute(0, 3, 1, 2)
    labels_tensor = torch.tensor(labels.numpy(), dtype=torch.long)

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

class ColorectalHistDatasetSemi(Dataset):
    def __init__(self, images, labels, transform_w=None, transform_s=None):
        self.images = images
        self.labels = labels
        self.transform_w = transform_w
        self.transform_s = transform_s

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx].clone().detach()
        label = self.labels[idx].clone().detach()

        if self.transform_w:
            image_w = self.transform_w(image)
        else:
            image_w = image

        if self.transform_s:
            image_s = self.transform_s(image)
        else:
            image_s = image


        return image_w, image_s, label

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

# Transformation for supervised learning
transform = transforms.Compose([

    transforms.ToPILImage(),  # Convert numpy array to PIL Image to apply transforms
    transforms.RandomHorizontalFlip(p=0.5),  # Apply horizontal flip with 50% probability
    transforms.RandomVerticalFlip(p=0.5),    # Apply vertical flip with 50% probability
    # Add any additional transformations here (e.g., resizing, normalization)
    transforms.RandomResizedCrop(size=(150, 150), antialias=True),
    transforms.ToTensor(),  # Convert PIL Image back to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = ColorectalHistDataset(images_tensor, labels_tensor, transform)
train_subset, test_subset = stratified_split(dataset)


# In[9]:


# Transformation for semi-supervised learning
transform_w = transforms.Compose([
    transforms.ToPILImage(),  # Convert numpy array to PIL Image to apply transforms
    transforms.RandomHorizontalFlip(p=0.5),  # Apply horizontal flip with 50% probability
    transforms.RandomVerticalFlip(p=0.5),    # Apply vertical flip with 50% probability
    transforms.ToTensor(),  # Convert PIL Image back to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_s = transforms.Compose([
    transforms.ToPILImage(),  # Convert numpy array to PIL Image to apply transforms
    transforms.RandomHorizontalFlip(p=0.5),  # Apply horizontal flip with 50% probability
    transforms.RandomVerticalFlip(p=0.5),    # Apply vertical flip with 50% probability
    RandAugmentMC(n=2, m=10),
    transforms.ToTensor(),  # Convert PIL Image back to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# In[15]:


# The base learner

class CRCClassifier(nn.Module):
    def __init__(self, num_classes=8):
        super(CRCClassifier, self).__init__()
        self.num_classes = num_classes
        resnet18 = models.resnet18(pretrained=True)
        resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)
        self.conv_layers = resnet18

    def forward(self, x):
        return self.conv_layers(x)

model = CRCClassifier().to(device)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(seed)

# Settings for training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
num_epochs = 30
eval_step = 200
threshold = 0.95
batch_size = 40
ratio = 5
lambda_u = 1

# Labeled instances selection
dataset = ColorectalHistDatasetSemi(images_tensor, labels_tensor, transform_w, transform_s)
train_subset, test_subset = stratified_split(dataset)
unlabeled_subset, labeled_subset = stratified_split(train_subset, test_size=0.02)
labeled_loader = DataLoader(labeled_subset, batch_size=batch_size, shuffle=True, num_workers=2)
unlabeled_loader = DataLoader(unlabeled_subset, batch_size=batch_size*ratio, shuffle=True, num_workers=2)
val_loader = DataLoader(test_subset, batch_size=40, shuffle=False)

def interleave(x, batch_size):
    s = list(x.shape)
    return x.reshape([-1, batch_size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def train_fixmatch(model, labeled_loader, unlabeled_loader, optimizer, num_epochs, eval_step, threshold, lambda_u):
    model.train()
    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)
    running_loss = 0.0
    for epoch in range(num_epochs):
        loss_s = 0.0
        loss_u = 0.0
        for batch_idx in range(eval_step):
            try:
                inputs_x, _, targets_x = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                inputs_x, _, targets_x = next(labeled_iter)

            try:
                inputs_u_w, inputs_u_s, _ = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                inputs_u_w, inputs_u_s, _ = next(unlabeled_iter)

            inputs_x = inputs_x.to(device)
            targets_x = targets_x.to(device)
            inputs_u_w = inputs_u_w.to(device)
            inputs_u_s = inputs_u_s.to(device)

            # Forward pass
            optimizer.zero_grad()
            inputs = interleave(torch.cat((inputs_x, inputs_u_w, inputs_u_s)), ratio*2+1).to(device)
            logits = model(inputs)
            logits = de_interleave(logits, ratio*2+1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits

            # Compute the loss for labeled data
            Lx = criterion(logits_x, targets_x)
            loss_s += Lx.item() * inputs_x.size(0)

            # Compute the pseudo-labels and loss for unlabeled data
            probs = torch.softmax(logits_u_w, dim=-1)
            max_probs, targets_u = torch.max(probs, dim=-1)
            mask = max_probs.ge(threshold).bool()

            Lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()

            loss_u += Lu.item() * mask.sum()

            # Total loss
            loss = Lx + lambda_u * Lu

            # Backward and optimize
            loss.backward()
            optimizer.step()
            model.zero_grad()

        loss_s = loss_s / (eval_step * batch_size)
        loss_u = loss_u / (eval_step * batch_size * ratio)

        print(f'Supervised Loss: {loss_s}, Unsupervised Loss: {loss_u}')

        # Validation phase
        model.eval()  # Set model to evaluate mode
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs_w, inputs_s, labels in val_loader:
                inputs = inputs_w.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(test_subset)
        val_acc = val_corrects.double() / len(test_subset)
        print(f'Validation Loss: {val_loss}, Acc: {val_acc}')


def train_fixmatch_multistep(model, labeled_loader, unlabeled_loader, optimizer, num_epochs, eval_step, threshold, lambda_u=1): 
    model.train()
    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)
    best_acc = 0.0
    for epoch in range(num_epochs):
        loss_s = 0.0
        loss_u = 0.0
        for batch_idx in range(eval_step):
            try:
                inputs_x, _, targets_x = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                inputs_x, _, targets_x = next(labeled_iter)

            try:
                inputs_u_w, inputs_u_s, _ = next(unlabeled_iter)
                if len(inputs_u_w) < batch_size*ratio:
                    unlabeled_iter = iter(unlabeled_loader)
                    inputs_u_w, inputs_u_s, _ = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                inputs_u_w, inputs_u_s, _ = next(unlabeled_iter)

            inputs_x = inputs_x.to(device)
            targets_x = targets_x.to(device)
            inputs_u_w = inputs_u_w.to(device)
            inputs_u_s = inputs_u_s.to(device)
            # print(inputs_x.shape[0], inputs_u_w.shape[0], inputs_u_s.shape[0])

            # Forward pass
            optimizer.zero_grad()
            inputs = interleave(torch.cat((inputs_x, inputs_u_w, inputs_u_s)), ratio*2+1).to(device)
            logits = model(inputs)
            logits = de_interleave(logits, ratio*2+1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits

            # Compute the loss for labeled data
            Lx = criterion(logits_x, targets_x)
            loss_s += Lx.item() * inputs_x.size(0)

            # Compute the pseudo-labels and loss for unlabeled data
            probs = torch.softmax(logits_u_w, dim=-1)
            max_probs, targets_u = torch.max(probs, dim=-1)
            mask = max_probs.ge(threshold).bool()

            Lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()

            loss_u += Lu.item() * mask.sum()

            # Total loss
            loss = Lx + lambda_u * Lu

            # Backward and optimize
            loss.backward()
            optimizer.step()
            model.zero_grad()

        loss_s = loss_s / (eval_step * batch_size)
        loss_u = loss_u / (eval_step * batch_size * ratio)

        print(f'Supervised Loss: {loss_s}, Unsupervised Loss: {loss_u}')

        # Validation phase
        model.eval()  # Set model to evaluate mode
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs_w, inputs_s, labels in val_loader:
                inputs = inputs_w.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(test_subset)
        val_acc = val_corrects.double() / len(test_subset)
        print(f'Validation Loss: {val_loss}, Acc: {val_acc}')
        if val_acc > best_acc:
            best_acc = val_acc
        
    return best_acc

def active_sampling(model, unlabeled_subset, strategy = 'uncertainty', forward_passes = 100):
    model.eval()
    with torch.no_grad():
        all_predictions = []
        unlabeled_loader = DataLoader(unlabeled_subset, batch_size=batch_size, shuffle=False, num_workers=2)
        if strategy == 'uncertainty':
            for _, inputs_s, _ in unlabeled_loader:
                mean_predictions = []
                for _ in range(forward_passes):
                    inputs = inputs_s.to(device)
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
        elif strategy == 'diversity':
            for inputs_w, _, _ in unlabeled_loader:
                mean_predictions = []
                for _ in range(forward_passes):
                    inputs = inputs_s.to(device)
                    outputs = model(inputs)
                    probs = torch.softmax(outputs, dim=-1)
                    mean_predictions.append(probs.unsqueeze(0))
                mean_predictions = torch.cat(mean_predictions)
                mean_predictions = torch.mean(mean_predictions, 0)
                all_predictions.append(mean_predictions)
            all_predictions = torch.cat(all_predictions)
            max_probs, targets_u = torch.max(all_predictions, dim=-1)
            mask = max_probs.ge(threshold).bool()
            
        else:
            pass
    
    return indices
                

def active_sampling(model, unlabeled_subset, strategy = 'uncertainty', forward_passes = 100):
    model.eval()
    with torch.no_grad():
        unlabeled_loader = DataLoader(unlabeled_subset, batch_size=batch_size, shuffle=False, num_workers=2)
        if strategy == 'uncertainty':
            all_predictions = []
            for _, inputs_s, _ in unlabeled_loader:
                mean_predictions = []
                for _ in range(forward_passes):
                    inputs = inputs_s.to(device)
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
        elif strategy == 'diversity':
            # Compute the pairwise similarity between predictions
            for inputs_w, _, _ in unlabeled_loader:
                mean_predictions = []
                for _ in range(forward_passes):
                    inputs = inputs_s.to(device)
                    outputs = model(inputs)
                    probs = torch.softmax(outputs, dim=-1)
                    mean_predictions.append(probs.unsqueeze(0))
                mean_predictions = torch.cat(mean_predictions)
                mean_predictions = torch.mean(mean_predictions, 0)
                all_predictions.append(mean_predictions)
            all_predictions = torch.cat(all_predictions)
            # Compute the pairwise similarity between predictions
            similarity_matrix = torch.zeros((len(all_predictions), len(all_predictions)))
            for i in range(len(all_predictions)):
                for j in range(i+1, len(all_predictions)):
                    similarity_matrix[i, j] = torch.cosine_similarity(all_predictions[i], all_predictions[j])
                    similarity_matrix[j, i] = similarity_matrix[i, j]
            
            # Select the indices with the highest diversity
            selected_indices = []
            while len(selected_indices) < batch_size:
                # Find the index with the lowest average similarity to the already selected indices
                min_similarity = float('inf')
                min_index = None
                for i in range(len(all_predictions)):
                    if i not in selected_indices:
                        similarity = torch.mean(similarity_matrix[i, selected_indices])
                        if similarity < min_similarity:
                            min_similarity = similarity
                            min_index = i
                
                # Add the selected index to the list and update the similarity matrix
                selected_indices.append(min_index)
                similarity_matrix = torch.cat((similarity_matrix[:min_index, :], similarity_matrix[min_index+1:, :]), dim=0)
                similarity_matrix = torch.cat((similarity_matrix[:, :min_index], similarity_matrix[:, min_index+1:]), dim=1)
        elif strategy == 'consistency':
            all_predictions_s = []
            all_predictions_w = []
            for inputs_w, inputs_s, _ in unlabeled_loader:
                mean_predictions_w = []
                mean_predictions_s = []
                for _ in range(forward_passes):
                    inputs_s = inputs_s.to(device)
                    inputs_w = inputs_w.to(device)
                    outputs_s = model(inputs_s)
                    outputs_w = model(inputs_w)
                    probs_s = torch.softmax(outputs_s, dim=-1)
                    probs_w = torch.softmax(outputs_w, dim=-1)
                    mean_predictions_s.append(probs_s.unsqueeze(0))
                    mean_predictions_w.append(probs_w.unsqueeze(0))
                mean_predictions_s = torch.cat(mean_predictions_s)
                mean_predictions_s = torch.mean(mean_predictions_s, 0)
                mean_predictions_w = torch.cat(mean_predictions_w)
                mean_predictions_w = torch.mean(mean_predictions_w, 0)
                
                all_predictions_s.append(mean_predictions_s)
                all_predictions_w.append(mean_predictions_w)
            all_predictions_s = torch.cat(all_predictions_s)
            all_predictions_w = torch.cat(all_predictions_w)
            # Compute the pairwise similarity between predictions
            scores = -torch.cosine_similarity(all_predictions_s, all_predictions_w, dim = -1)
            _, indices = torch.topk(scores, batch_size)
        elif strategy == 'combined':
            all_predictions_s = []
            all_predictions_w = []
            for inputs_w, inputs_s, _ in unlabeled_loader:
                mean_predictions_w = []
                mean_predictions_s = []
                for _ in range(forward_passes):
                    inputs_s = inputs_s.to(device)
                    inputs_w = inputs_w.to(device)
                    outputs_s = model(inputs_s)
                    outputs_w = model(inputs_w)
                    probs_s = torch.softmax(outputs_s, dim=-1)
                    probs_w = torch.softmax(outputs_w, dim=-1)
                    mean_predictions_s.append(probs_s.unsqueeze(0))
                    mean_predictions_w.append(probs_w.unsqueeze(0))
                mean_predictions_s = torch.cat(mean_predictions_s)
                mean_predictions_s = torch.mean(mean_predictions_s, 0)
                mean_predictions_w = torch.cat(mean_predictions_w)
                mean_predictions_w = torch.mean(mean_predictions_w, 0)
                
                all_predictions_s.append(mean_predictions_s)
                all_predictions_w.append(mean_predictions_w)
            all_predictions_s = torch.cat(all_predictions_s)
            all_predictions_w = torch.cat(all_predictions_w)
            # Compute the pairwise similarity between predictions
            consistency_scores = torch.cosine_similarity(all_predictions_s, all_predictions_w, dim = -1)
            safe_probabilities = all_predictions_s.clamp(min=1e-9)
            uncertainty_scores = -torch.sum(safe_probabilities * torch.log(safe_probabilities), dim = 1)
            combined_scores = consistency_scores * uncertainty_scores
            _, indices = torch.topk(combined_scores, batch_size)
        else:
            pass
    
    return indices
                

for i in range(5, 10):
    set_seed(seed)
    model = CRCClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    curr_train_data_portion = 0.01*i
    print(f'Currently {curr_train_data_portion*100}% of the train data is being used')
    if i == 5:
        unlabeled_subset, labeled_subset = stratified_split(train_subset, test_size=curr_train_data_portion)
        labeled_loader = DataLoader(labeled_subset, batch_size=batch_size, shuffle=True, num_workers=2)
        unlabeled_loader = DataLoader(unlabeled_subset, batch_size=batch_size*ratio, shuffle=True, num_workers=2)
        acc = train_fixmatch_multistep(model, labeled_loader, unlabeled_loader, optimizer, num_epochs, eval_step, threshold, lambda_u=1)
        #torch.save(model.state_dict(), 'semi_5_percents_combined.pth')
        indices = active_sampling(model, unlabeled_subset, 'combined')
        labeled_subset = ConcatDataset([labeled_subset,Subset(unlabeled_subset, indices.tolist())])
        all_indices = torch.arange(len(unlabeled_subset)).to(device)
        mask = ~torch.isin(all_indices, indices)
        indices_to_keep = all_indices[mask]
        unlabeled_subset = Subset(dataset, indices_to_keep.tolist())
    else:
        labeled_loader = DataLoader(labeled_subset, batch_size=batch_size, shuffle=True, num_workers=2)
        unlabeled_loader = DataLoader(unlabeled_subset, batch_size=batch_size*ratio, shuffle=True, num_workers=2)
        acc = train_fixmatch_multistep(model, labeled_loader, unlabeled_loader, optimizer, num_epochs, eval_step, threshold, lambda_u=1)
        indices = active_sampling(model, unlabeled_subset, 'combined')
        labeled_subset = ConcatDataset([labeled_subset,Subset(unlabeled_subset, indices.tolist())])
        all_indices = torch.arange(len(unlabeled_subset)).to(device)
        mask = ~torch.isin(all_indices, indices)
        indices_to_keep = all_indices[mask]
        unlabeled_subset = Subset(dataset, indices_to_keep.tolist())
    print(f'best_acc:{acc}')
    with open('semi_val_accs_act_combined_0.05_3.txt','a+') as f:
        f.write(f'{acc}\n')
    f.close()

