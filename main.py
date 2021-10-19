    
    
from model import CNN
from utils import import_preprocess_test
import torch
# from torchvision import transforms, datasets
# from torch.utils import data
# import torch.nn as nn
# import matplotlib.pyplot as plt

if __name__ == "__main__":
    path = "/Users/danielpetterson/GitHub/PytorchImageClassification/Data/traindata"
    batch_size = 10
    test_data = import_preprocess_test(path, batch_size)
    model = CNN()
    model.load_state_dict(torch.load("savedModel.pth"))
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    model.eval()