# Pytorch Image Classification

These scripts are used to train and evaluate the convolutional neural network defined in `model.py`.


# HOW TO USE

Ensure that the you have Python 3 and the required libraries installed on your system and that `train.py`, `test.py`, `model.py` and `model.pth` are in the same directory.

model.pth is the pretrained model and running `train.py` will overwrite the model parameters.
To run evaluate the model on new data, place the data in class defining subfolders within the testdata folder and enter the following into your terminal:

$ python `test.py` (on Windows)

or:

$ python3 `test.py` (on Mac or Linux)

You can run the training script in a similar manner.


## 1.1 Output

train.py saves a copy of the final model weights to model.pth and displays accuracy for each k-fold and epoch.
test.py prints the accuracy of the model on the new data.