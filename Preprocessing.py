import os, os.path
from glob import glob
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, datasets
import torch.nn as nn

# os.getcwd()
# sys.path.append('/Users/danielpetterson/miniforge3/lib/python3.9/site-packages')

data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

path = "/Users/danielpetterson/GitHub/PytorchImageClassification/Data/traindata"
dataset = datasets.ImageFolder(root=path, transform=data_transform)


# Hyperparameters
num_epochs = 50
num_classes = 3
batch_size = 10
learning_rate = 0.001


dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU())
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU())
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop_out = nn.Dropout()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(100352,512),
            nn.ReLU(),
            nn.Linear(512,64),
            nn.ReLU(),
            nn.Linear(64, 3))

    def forward(self, input):
        out = self.conv1(input)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.pool2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out


model=CNN()


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(dataset_loader)
loss_list = []
acc_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(dataset_loader):
        # Run the forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),(correct / total) * 100))