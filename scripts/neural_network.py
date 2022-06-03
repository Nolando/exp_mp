#!/usr/bin/env python3
# This script detects human faces and matches them with a trained
# neural network to determine if they are a target. The coordinate
# of the person's face is also calculated.

# Packages
import os
#import rospy
import cv2 as cv
import pandas as pd
import numpy as np
import glob
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torchvision
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import PIL
from PIL import Image
from torchvision import datasets
from torchvision.transforms import ToTensor
#from scipy.misc import face


################################################################################
def neural_network():

    ################################ Define some custom transforms to apply to the image ####################
    custom_transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Training data path
    trainingPath = '/home/aidanstapleton/faces/'

    # Add the training images (selfies) to the training set data list
    imageDirectories = []
    [imageDirectories.extend(glob.glob(trainingPath + '*.jpg'))]
    trainingImages = [cv.imread(currentImage) for currentImage in imageDirectories]

    # Get the image file names
    imageNames = []
    for imageName in os.listdir(trainingPath):
        
        imageNames.append(imageName)


    ####### Make a list of labels for each person of interest in the data set plus a category for 'other' ########
    # List of image labels (must correspond to the order of the images)
    nameLabels = ['Aidan', 'Aidan', 'Aidan', 'Aidan', 'Aidan', 'Aidan', 'Aidan', 'Aidan', 'Kieran', 'Kieran', 'Kieran', 'Kieran', 'Kieran', 'Kieran', 'Kieran', 'Kieran']
    labels = []

    for image in os.listdir(trainingPath):

        # If image is of Aidan
        if image.__contains__('aidan'):
            labels.append(0)

        # If image is of Kieran
        elif image.__contains__('Kieran'):
            labels.append(1)

        # If the image is a random person
        else:
            labels.append(2)

    print('Number of Aidans in training set =', labels.count(0))
    print('Number of Kierans in training set =', labels.count(1))
    print('Number of randoms in training set =', labels.count(2))

    # Assign the training dataset to faces file
    # trainset = torchvision.datasets.CIFAR10(root='faces', 
    #                                         train=True,     # True for training set
    #                                         download=True, 
    #                                         transform=custom_transform)
    # testset = torchvision.datasets.CIFAR10(root='data', 
    #                                        train=False,    # False for test set
    #                                        download=True, 
    #                                        transform=custom_transform)

    # Subscribe to the face box node to get the next testing data
    #testset = face_box_sub
    testingImages = trainingImages[0]
 
    # Create the custom training data set of selfies
    class trainingImageDataset(Dataset):

        # Initialise the dataset object
        def __init__(self, imagesDirectory, imageNames, labels, transform=None, target_transform=None):
            self.imagesDirectory = imagesDirectory
            self.labels = pd.DataFrame(labels)  # Convert list to a dataframe to use 'iloc'
            self.imageNames = pd.DataFrame(imageNames)
            self.transform = transform
            self.target_transform = target_transform

        # Get the length of the dataset
        def __len__(self):
            return len(self.labels)

        # Get a sample from the dataset
        def __getitem__(self, idx):

            # Get the image path of the current image in the training dataset
            img_path = os.path.join(self.imagesDirectory, self.imageNames.iloc[idx, 0])
            
            # Current image as an NP array
            #imageTensor = read_image(img_path)                  # Type = Tensor
            PIL_Image = PIL.Image.open(img_path)
            imageNpArray = np.array(PIL_Image)                  # Type = NP array

            # Label for the current image
            label = self.labels.iloc[idx, 0]
            #label = np.array(label)
            
            # Transform the current image and label into the correct formats
            if self.transform:                                  # Features need to be normalized tensors
                image = self.transform(imageNpArray)
            if self.target_transform:                           # Labels need to be one-hot encoded tensors
                label = self.target_transform(label)
            return image, label

    ################################ Dataloaders ###########################################################
    trainingDataset = trainingImageDataset(trainingPath, imageNames, labels, transform = custom_transform)


    trainloader = DataLoader(trainingDataset, batch_size=8, shuffle=True)
    testloader = DataLoader(testingImages, batch_size=4, shuffle=False)

    ################################ Understanding the dataset #############################################
    ################################ This section just prints info #########################################
    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 
    #            'dog', 'frog', 'horse', 'ship', 'truck')
    classes = ('Aidan', 'Kieran', 'Random Person')
    
    # print number of samples                                                       
    print("Number of training samples is {}".format(len(trainingDataset)))
    #print("Number of test samples is {}".format(len(testingSet)))

    # iterate through the training set print useful information
    dataiter = iter(trainloader)
    images, labels = dataiter.next()    # this gather one batch of data

    print("Batch size is {}".format(len(images)))
    print("Size of each image is {}".format(images[0].shape))

    print("The labels in this batch are: {}".format(labels))
    # print("These correspond to the classes: {}, {}, {}, {}, {}, {}, {}, {}".format(
    #     classes[labels[0]], classes[labels[1]],
    #     classes[labels[2]], classes[labels[3]],
    #     classes[labels[4]], classes[labels[5]],
    #     classes[labels[6]], classes[labels[7]]))

    # plot images of the batch
    fig, ax = plt.subplots(1, len(images))
    for id, image in enumerate(images):
        # convert tensor back to numpy array for visualization
        ax[id].imshow((image / 2 + 0.5).numpy().transpose(1,2,0))
        ax[id].set_title(classes[labels[id]])

    ################################ Define the neural network (NN) ########################################
    class Network(nn.Module):

        def __init__(self):

            # Define the NN layers
            self.output_size = 2   # 10 classes

            super(Network, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)             # 2D convolution
            self.pool = nn.MaxPool2d(2, 2)              # max pooling
            self.conv2 = nn.Conv2d(6, 16, 5)            # 2D convolution
            #self.fc1 = nn.Linear(16 * 5 * 5, 120)       # Fully connected layer
            self.fc1 = nn.Linear(16 * 355 * 266 , 120)       # Fully connected layer
            self.fc2 = nn.Linear(120, 84)               # Fully connected layer
            self.fc3 = nn.Linear(84, self.output_size)  # Fully connected layer

        def forward(self, x):
            
            # Define the forward pass
            x = self.pool(functional.relu(self.conv1(x)))
            x = self.pool(functional.relu(self.conv2(x)))
            #x = x.view(-1, 16 * 5 * 5)
            print('Size of x: ', x.size(0))
            x = x.view(-1, 16 * 355 * 266)          # x.view(batch_size, channels * height * width)
            x = functional.relu(self.fc1(x))
            x = functional.relu(self.fc2(x))
            x = self.fc3(x)
            return x
            
    net = Network()

    ################################ Define the loss function and optimizer ################################
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    ################################ Train the neural network with the training data #######################

    # Number of epochs (cycles)
    epochCount = 5

    # Loop through each epoch
    for epoch in range(epochCount):    # we are using 5 epochs. Typically 100-200
        running_loss = 0.0

    for i, data in enumerate(trainloader, 0):

        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # Perform forward pass and predict labels
        predicted_labels = net(inputs)
    
        print('Predicted Labels Size =', predicted_labels.shape)
        print(predicted_labels)
        print('Labels Size =', labels.shape)

        # Calculate loss
        loss = criterion(predicted_labels, labels)

        # Perform back propagation and compute gradients
        loss.backward()
        
        # Take a step and update the parameters of the network
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # if i % 2000 == 1999:    # print every 2000 mini-batches
        print('Epoch: %d, %5d loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
        running_loss = 0.0

    print('Finished Training.')
    torch.save(net.state_dict(), "facesTrained.pth")
    print('Saved model parameters to disk.')

    ################################ Use the trained neural network to identify the target ##################
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    fig, ax = plt.subplots(1, len(images))
    for id, image in enumerate(images):

        # convert tensor back to numpy array for visualization
        ax[id].imshow((image / 2 + 0.5).numpy().transpose(1,2,0))
        ax[id].set_title(classes[labels[id]])

    plt.show()

    # Predict the output using the trained neural network
    outputs = net(images)

    # Normalize the outputs using the Softmax function so that
    # we can interpret it as a probability distribution.
    sm = nn.Softmax(dim=1)      
    sm_outputs = sm(outputs)

    # For each output the prediction with the highest probability
    # is the predicted label
    probs, index = torch.max(sm_outputs, dim=1)
    for p, i in zip(probs, index):
        print('True label {0}, Predicted label {0} - {1:.4f}'.format(classes[i], p))


if __name__ == '__main__':
    
    neural_network()
    
