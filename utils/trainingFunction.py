
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
def trainingFunction(model, train_loader,test_loader, epochs, criterion, optimizer):
  train_loss = []
  test_loss = []
  train_correct =[]
  test_correct = []

 #For loop of Epochs
  for i in range(epochs):
    train_corr = 0
    test_corr = 0
    total_samples = 0
    test_total_samples = 0


  #Train
    for b,(X_train, Y_train) in enumerate(train_loader):
      b += 1 #start our batches at 1
      y_pred = model(X_train) #get the predicted values from the training set. Not flaattedned 2D
      loss = criterion(y_pred, Y_train) #How off are we? Compare the prediction to the correct answers in Y_train
      predicted = torch.max(y_pred.data, 1)[1] #we want to add up the correct number of prediciton. Indexed of the first point
      batch_corr = (predicted == Y_train).sum() #we want to know how many we got correct.
      train_corr += batch_corr #keep track as we go aloing in training
      total_samples += Y_train.size(0)

    #Update the parameters
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      if b % 600 == 0:
        print(f'Epoch: {i+1}, Batch {b}, Loss {loss.item(): .4f}')

  #append train loss and train correct into the list
    train_loss.append(loss.item())
    train_correct.append(train_corr.item())

  #Test
    with torch.no_grad():
      for b,(X_test, Y_test) in enumerate(test_loader):
        y_value = model(X_test)
        loss = criterion(y_value, Y_test)
        predicted = torch.max(y_value.data, 1)[1]
        test_corr += (predicted == Y_test).sum()
        test_total_samples += Y_test.size(0)

 #append test loss and train correct into the list
    test_loss.append(loss.item())
    test_correct.append(test_corr.item())
    train_accuracy = 100 * train_corr.item() / total_samples
    test_accuracy = 100 * test_corr.item() / test_total_samples
    print(f"Epoch {i + 1} — Train Accuracy: {train_accuracy:.2f}%")
    print(f"Epoch {i + 1} — Test Accuracy: {test_accuracy:.2f}%")


  return train_loss, test_loss, train_correct, test_correct
  