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
        transforms.Normalize()
    ])

path = "/Users/danielpetterson/GitHub/PytorchImageClassification/Data/traindata"
dataset = datasets.ImageFolder(root=path, transform=data_transform)

dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=2)


# os.chdir(path)
# image_paths = glob('./**/*.jpg', recursive=True)

# X = []
# for img in image_paths:
#     img_pixels = Image.open(img).getdata()
#     X.append(img_pixels)

# # Convert to array
# X=np.asarray(X)

# # Normalization
# # X=X/255

# # One-hot encoded target variables
# y = np.array([[1,0,0],[0,1,0],[0,0,1]])
# y = np.repeat(y, [cherry_count,straw_count,tomato_count], axis=0)




# Hyperparameters
num_epochs = 5
num_classes = 3
batch_size = 100
learning_rate = 0.001

# train_loader = torch.utils.data.DataLoader(dataset=(X,y), batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU())
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU())
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 3)

    def forward(self, input):
        out = self.conv1(input)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.pool2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
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