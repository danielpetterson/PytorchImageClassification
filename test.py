import torch
from torchvision import transforms, datasets
# from torchvision import transforms, datasets
# from torch.utils import data
# import torch.nn as nn
# import matplotlib.pyplot as plt

# Define model
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

def import_preprocess_test(path):\

    # Define data transformations
    data_transform = transforms.Compose([
        transforms.Resize([230, 230]),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])

    # Import dataset and apply transformation
    dataset = datasets.ImageFolder(root=path, transform=data_transform)

    return dataset

if __name__ == "__main__":
    path = "/Users/danielpetterson/GitHub/PytorchImageClassification/Data/traindata"
    # Hyperparameters
    num_epochs = 3
    num_classes = 3
    batch_size = 10
    learning_rate = 0.001

    # Instantiate model
    model = CNN()

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    test_data = import_preprocess_test(path, batch_size)
    
    model.load_state_dict(torch.load("savedModel.pth"))
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    model.eval()
    for data, labels in test_data:
            target = model(data)
            loss = criterion(target,labels)
            valid_loss = loss.item()