import numpy as np
import torch
import torch.utils as ut
from sklearn.model_selection import KFold
from utils import import_preprocess
from model import MLP, CNN
import matplotlib.pyplot as plt


# Set seed for reproducability
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


# Hyperparameters
num_epochs = 3
num_classes = 3
batch_size = 10
learning_rate = 0.001
k_folds = 5


def train(data, model=CNN(), criterion=None, optimizer=None, fold_num=5, batch_size=10, num_epochs=3, learning_rate=0.001):
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)
    # Instantiate model
    model=model
    print(model)

    for fold_num, (train, val) in enumerate(kfold.split(data)):
        print(f'Fold {fold_num+1}:')

        # Sample elements randomly without replacement.
        train_subsampler = ut.data.SubsetRandomSampler(train)
        val_subsampler = ut.data.SubsetRandomSampler(val)

        # Define dataloaders for training and validation
        train_data = ut.data.DataLoader(data, batch_size=batch_size, sampler=train_subsampler)
        val_data = ut.data.DataLoader(data, batch_size=batch_size, sampler=val_subsampler)

        # Loss and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Check for GPU
        if torch.cuda.is_available():
            print("GPU Available")
            model = model.cuda()
            criterion = criterion.cuda()

        train_loss_hist = []
        val_loss_hist = []
        acc = {}

        for epoch in range(num_epochs):

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
                valid_loss = loss.item()
                # print(labels)
                _, predicted = torch.max(outputs.data, 1)
                # print(predicted)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            val_loss_hist.append(valid_loss/len(val_data))
            accuracy = (correct / total) * 100.0
            acc[fold_num] = accuracy
            print('Accuracy for fold %d: %d %%' % (fold_num+1, 100.0 * correct / total))


            print(f'Epoch {epoch+1} \n Training Loss: {train_loss / (len(train_data) * num_epochs)} \n Validation Loss: {valid_loss / len(val_data)}')
            if min_valid_loss > valid_loss:
                print(' Saving model due to validation loss decrease')
                min_valid_loss = valid_loss
                # Saving model state
                path = f'./savedModel_fold_{fold_num}.pth'
                torch.save(model.state_dict(), path)
        # # Print fold results
        print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
        sum = 0.0
        for key, value in acc.items():
            print(f'Fold {key}: {value} %')
            sum += value
        print(f'Average: {sum/len(acc.items())} %')

        # plt.plot(train_loss_hist)
        # plt.plot(val_loss_hist)
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.legend(['Training','Validation'])
        # plt.title('Loss over time')

        # plt.show()

if __name__ == "__main__":
    # Import training and validation datasets
    path = "/Users/danielpetterson/GitHub/PytorchImageClassification/Data/traindata"
    data = import_preprocess(path, batch_size)
    train(data=data, model=MLP())

    

    