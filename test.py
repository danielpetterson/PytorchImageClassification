import os
import torch
from torchvision import transforms, datasets
from model import CNN
import torch.utils as ut


if __name__ == "__main__":
    # Instantiate model and load weights
    model = CNN()
    location = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    model.load_state_dict(torch.load(os.path.join(location, "model.pth")))

        # Define data transformations
    data_transform = transforms.Compose([
        # Scale to same size as training images
        transforms.Resize([230, 230]),
        transforms.ToTensor(),
        # Normalise to same values as training data
        transforms.Normalize([0.5],[0.5])
    ])

    # Import dataset and apply transformation
    path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), "testdata"))
    test_data = datasets.ImageFolder(root=path, transform=data_transform)
    test_loader = ut.data.DataLoader(test_data, batch_size=32)
    dataiter = iter(test_loader)
    data, labels = dataiter.next()
    
    # Calculate accuracy on test set
    correct = 0
    total = 0
    model.eval()
    outputs = model(data)
    _, predicted = torch.max(outputs.data, 1)
    total += outputs.size(0)
    correct += (predicted == labels).sum().item()
    accuracy = (correct / total) * 100.0
    print('Accuracy on test set: %d %%' % (accuracy))