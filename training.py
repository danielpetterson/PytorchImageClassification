import numpy as np
import torch
from torch.utils import data
from utils import import_preprocess
from model import CNN
import matplotlib.pyplot as plt


# Set seed for reproducability
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


# Hyperparameters
num_epochs = 10
num_classes = 3
batch_size = 10
learning_rate = 0.001

if __name__ == "__main__":
    # Import training and validation datasets
    path = "/Users/danielpetterson/GitHub/PytorchImageClassification/Data/traindata"
    train_data, val_data = import_preprocess(path, 0.75, batch_size)

    # Instantiate model
    model=CNN()
    print(model)

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Check for GPU
    if torch.cuda.is_available():
        print("GPU Available")
        model = model.cuda()
        criterion = criterion.cuda()

    min_valid_loss = np.inf
    train_loss_hist = []
    val_loss_hist = []

    for epoch in range(num_epochs):

        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        for data, labels in train_data:
            
            optimizer.zero_grad()
            target = model(data)
            loss = criterion(target,labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss_hist.append(train_loss/len(train_data))
        
        model.eval()
        for data, labels in val_data:
            target = model(data)
            loss = criterion(target,labels)
            valid_loss = loss.item()
        val_loss_hist.append(valid_loss/len(val_data))


        print(f'Epoch {epoch+1} \n Training Loss: {train_loss / len(train_data)} \n Validation Loss: {valid_loss / len(val_data)}')
        if min_valid_loss > valid_loss:
            print(' Saving model due to validation loss decrease')
            min_valid_loss = valid_loss
            # Saving model state
            torch.save(model.state_dict(), 'savedModel.pth')

    plt.plot(train_loss_hist)
    plt.plot(val_loss_hist)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training','Validation'])
    plt.title('Loss over time')
    
    plt.show()