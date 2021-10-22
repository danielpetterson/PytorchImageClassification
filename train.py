import os
import numpy as np
import torch
import torch.nn as nn
from model import CNN
import torch.utils as ut
from sklearn.model_selection import KFold
from torchvision import transforms, datasets

# Set seed for reproducability
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# Hyperparameters
num_epochs = 2
num_classes = 3
batch_size = 32
learning_rate = 0.001
k_folds = 5

def import_preprocess(path):

    # Define data transformations
    data_transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomResizedCrop(230),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])

    # Import dataset and apply transformation
    dataset = datasets.ImageFolder(root=path, transform=data_transform)

    return dataset


def train(data, model=CNN(), criterion=nn.CrossEntropyLoss(), k_folds=k_folds, batch_size=batch_size, epochs=num_epochs, lr=learning_rate):

        # Check for GPU
    if torch.cuda.is_available():
        print("GPU Available")
        model = model.cuda()
        criterion = criterion.cuda()
    else:
        model = model
        criterion = criterion
    
    # Print model structure
    print(model)

    # Define optimizer and learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)

    for fold_num, (train, val) in enumerate(kfold.split(data)):
        print(f'Fold {fold_num+1}:')

        # Sample elements randomly without replacement.
        train_subsampler = ut.data.SubsetRandomSampler(train)
        val_subsampler = ut.data.SubsetRandomSampler(val)

        # Define dataloaders for training and validation
        train_data = ut.data.DataLoader(data, batch_size=batch_size, sampler=train_subsampler)
        val_data = ut.data.DataLoader(data, batch_size=batch_size, sampler=val_subsampler)

        train_loss_hist = []
        val_loss_hist = []
        acc = {}

        for epoch in range(epochs):

            train_loss = 0.0
            valid_loss = 0.0
            min_valid_loss = np.inf

            model.train()
            for dataset, labels in train_data:
                
                optimizer.zero_grad()
                outputs = model(dataset)
                loss = criterion(outputs,labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss_hist.append(train_loss/len(train_data))
            
            model.eval()
            correct = 0.0
            total = 0.0
            for dataset, labels in val_data:
                outputs = model(dataset)
                loss = criterion(outputs,labels)
                valid_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            val_loss_hist.append(valid_loss/len(val_data))
            accuracy = (correct / total) * 100.0
            acc[fold_num] = accuracy
            print('Accuracy for fold %d: %d %%' % (fold_num+1, 100.0 * correct / total))


            print(f'Epoch {epoch+1} \n Training Loss: {train_loss / len(train_data)} \n Validation Loss: {valid_loss / len(val_data)}')
            if min_valid_loss > valid_loss:
                print(' Saving model due to validation loss decrease')
                min_valid_loss = valid_loss
                # Saving model state
                save_path = 'model.pth'
                # torch.save(model.state_dict(), save_path)
        # Print results
        sum = 0.0
        for key, value in acc.items():
            sum += value
        print(f'Average Accuracy over {k_folds} folds: {sum/len(acc.items())} %')

    return model

if __name__ == "__main__":
    # Import training and validation datasets
    path = path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), "testdata"))
    data = import_preprocess(path)
    train(data)


    

    