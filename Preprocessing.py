import os, os.path
from glob import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

# os.getcwd()
# sys.path.append('/Users/danielpetterson/miniforge3/lib/python3.9/site-packages')


path = "/Users/danielpetterson/GitHub/PytorchImageClassification/Data/traindata"
os.chdir(path)
image_paths = glob('./**/*.jpg', recursive=True)

# Count number in each category
cherry_count  = sum('cherry' in s for s in image_paths)
straw_count  = sum('strawberry' in s for s in image_paths)
tomato_count  = sum('tomato' in s for s in image_paths)

X = []
for img in image_paths:
    img_pixels = Image.open(img).getdata()
    X.append(img_pixels)


# One-hot encoded target variables
y = np.array([[1,0,0],[0,1,0],[0,0,1]])
y = np.repeat(y, [cherry_count,straw_count,tomato_count], axis=0)


def shuffle_split_data(X, y):
    split = np.random.choice(range(len(X)), int(0.7*len(X)))

    X_train = X[split]
    y_train = y[split]
    X_test =  X[~split]
    y_test = y[~split]

    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = shuffle_split_data(X, y)



# Hyperparameters
num_epochs = 5
num_classes = 3
batch_size = 100
learning_rate = 0.001

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123)

# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
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
total_step = len(train_loader)
loss_list = []
acc_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
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
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))