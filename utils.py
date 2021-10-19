
import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils import data

def import_preprocess(path, prop_train, batch_size):\
    # Import dataset
    dataset = datasets.ImageFolder(root=path)

    # Split into training and validation sets
    props = [int(np.ceil(prop_train*len(dataset))),
            int(np.floor((1-prop_train)*len(dataset)))]
    train, val = data.random_split(dataset, props)
    
    # Define transformation
    data_transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomResizedCrop(230),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])

    # Overwrite dataset, apply transformation and 
    dataset = datasets.ImageFolder(root=path, transform=data_transform)
    train, val = data.random_split(dataset, props)
    training_dataset_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2)
    validation_dataset_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=2)

    return training_dataset_loader, validation_dataset_loader

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
    test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return test_dataset_loader