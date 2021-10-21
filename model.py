import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(230*230*3,512),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(512,256),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(256,64),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(64, 3))

    def forward(self, input):
        out = self.mlp1(input)
        
        return out


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=12, stride=4, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=6, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(86528,512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 3))

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.fc(out)
        
        return out
