import torch.nn as nn


# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 224, kernel_size=12, stride=4, padding=0),
            nn.BatchNorm2d(224),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(224, 128, kernel_size=6, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(67712,512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64, 3))

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.fc(out)
        
        return out

# Define MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(224*224,512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64, 3))

    def forward(self, input):
        out = self.fc(input)
        
        return out