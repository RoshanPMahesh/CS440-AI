# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by James Soole for the Fall 2023 semester

"""
This is the main entry point for MP9. You should only modify code within this file.
The unrevised staff files will be used for all other files and classes when code is run, 
so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 1 the network should have the following architecture (in terms of hidden units):
        in_size -> h -> out_size , where  1 <= h <= 256
        
        We recommend setting lrate to 0.01 for part 1.

        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        
        # values from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        self.conv_layers = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=6, kernel_size=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(in_channels=6, out_channels=18, kernel_size=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(in_channels=18, out_channels=32, kernel_size=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.full_layers = nn.Sequential(nn.Linear(288, 200), nn.ReLU(), nn.Linear(200, out_size))
        
        self.optimize = optim.SGD(self.parameters(), lr=0.01) # initializing the optimizer function
    

    def forward(self, x):
        """
        Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        x = x.view(-1, 3, 31, 31)   # used these values from a discussion in a piazza post
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.full_layers(x)
        return x
    

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """
        # returning loss.item() since it is just a single number
        self.optimize.zero_grad()
        yhat = self.forward(x)
        loss_value = self.loss_fn(yhat, y)
        loss_value.backward()
        self.optimize.step()
        return loss_value.item()


def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """ 
    Make NeuralNet object 'net'. Use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param epochs: an int, the number of epochs of training
    @param batch_size: size of each batch to train on. (default 100)

    This method *must* work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: list of floats containing the total loss at the beginning and after each epoch.
        Ensure that len(losses) == epochs.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    losses = []

    # Convert input arrays to PyTorch dataset
    train_dataset = get_dataset_from_arrays(train_set, train_labels)

    net = NeuralNet(lrate=0.01, loss_fn=nn.CrossEntropyLoss(),in_size=2883, out_size=4)  # 31 * 31 * 3 = 2883
    
    # Create DataLoader from the dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # used this website for understanding the dataloader: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    
    for iterations in range(epochs):
        epoch_loss = 0.0
        for batch in iter(train_loader):
            x = batch["features"]
            y = batch["labels"]
            epoch_loss += net.step(x, y)    # training part

        losses.append(epoch_loss / len(train_loader))

    output = net(torch.Tensor(dev_set))     # makes it a multi dimensional matrix
    throwaway, predicted = torch.max(output, 1)     # used this website: https://pytorch.org/docs/stable/generated/torch.max.html
    yhats = predicted.numpy()   # makes it a numpy array

    return losses, yhats, net

