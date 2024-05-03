#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
from torch.utils.data import Subset


# In[2]:


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


# In[15]:


# Load the dataset
dataset, info = tfds.load('colorectal_histology', with_info=True, as_supervised=True)
dataset = dataset['train'].batch(len(dataset['train']))


# In[16]:


# Transform the dataset into pytorch
for images, labels in dataset:
    images_tensor = torch.tensor(images.numpy(), dtype=torch.float)
    images_tensor = images_tensor.permute(0, 3, 1, 2)
    labels_tensor = torch.tensor(labels.numpy(), dtype=torch.long)


# In[17]:


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
            image1 = self.transform(image)
            image2 = self.transform(image)
        else:
            image1 = image
            image2 = image

        return image1, image2, label


# In[18]:


def stratified_split(dataset, test_size=0.2):

    labels = np.array([label for _, _, label in dataset])

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
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(size=120),
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
    transforms.GaussianBlur(kernel_size= 5, sigma=(0.1, 2.0)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = ColorectalHistDataset(images_tensor, labels_tensor, transform)
train_subset, test_subset = stratified_split(dataset)


# In[6]:


def info_nce_loss(self, features, batch_size, n_views = 2, temperature = 0.07):

    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels


# In[7]:


class SimCLR(nn.Module):
    def __init__(self, base_model, out_features=512, proj_dim=128):
        super(SimCLR, self).__init__()
        self.encoder = base_model(pretrained=False, num_classes=out_features)
        self.encoder.fc = nn.Identity()  # Remove the final layer
        self.projection_head = nn.Sequential(
            nn.Linear(out_features, 512), nn.ReLU(), nn.Linear(512, proj_dim)
        )

    def forward(self, x):
        features = self.encoder(x)
        projections = self.projection_head(features)
        return projections


# In[8]:


def nt_xent_loss(z_i, z_j, temperature = 0.07):
    cos_sim = torch.matmul(z_i, z_j.T) / temperature
    labels = torch.arange(z_i.size(0)).long().to(z_i.device)
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(cos_sim, labels)
    return loss


# In[19]:


# Initialize the model
base_model = models.resnet18
model = SimCLR(base_model).to(device)
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=2)

# Define the optimizer and learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_subset), eta_min=0, last_epoch=-1)


# In[21]:


epochs = 100
temperature = 0.07

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for (img1, img2, label) in train_loader:
        img1, img2 = img1.to(device), img2.to(device)

        optimizer.zero_grad()

        z_i = model(img1)
        z_j = model(img2)

        loss = nt_xent_loss(z_i, z_j, temperature)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')

    # Optionally save your model here with torch.save


# In[22]:


torch.save(model.state_dict(), 'unsupervised.pth')

