
import torch
from torchvision import transforms, datasets
# from torch.utils import data

def import_preprocess(path, batch_size):\

    # Define transformation
    data_transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomResizedCrop(230),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])

    # Import dataset and apply transformation
    dataset = datasets.ImageFolder(root=path, transform=data_transform)
    # dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    return dataset

def import_preprocess_test(path, batch_size):\

    # Define transformation
    data_transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomResizedCrop(230),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])

    # Import dataset and apply transformations
    test_dataset = datasets.ImageFolder(root=path, transform=data_transform)
    # test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return test_dataset