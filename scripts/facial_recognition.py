#!/usr/bin/env python3
# This script detects human faces and matches them with a trained
# neural network to determine if they are a target. The coordinate
# of the person's face is also calculated.

# Packages
import os
import rospy
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torchvision
import glob
from PIL import Image
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from sensor_msgs.msg import CompressedImage
import camera_functions

# Create the publisher
face_box_pub = rospy.Publisher("/django/eagle_eye/bounding_box_face", numpy_msg(Floats), queue_size=1)

# Initialise the face cascade classifier (done outside of function to try improve latency)
# + LATENCY fixed by setting camera subscriber queue size to 1
cascadePath = 'haarcascade_frontalface_default.xml'
faceCascade = cv.CascadeClassifier(cv.data.haarcascades + cascadePath)

#################################################################################
def camera_callback(frame):

    # log some info about the image topic
    # rospy.loginfo('facial_recognition\tCAMERA FRAME RECEIVED')

    # Convert the ROS image to OpenCV compatible image
    converted_frame = camera_functions.camera_bridge_convert(frame)

    # Get the face bounding box
    face_bounding_box = face_detect(converted_frame)      # np.ndarray if detected, tuple if empty

    # Test if the returned variable contains a detected face bounding box
    if face_bounding_box is not tuple():

        # Log message
        # rospy.loginfo('facial_recognition\tDETECTED FACE WITH BOX ' + np.array2string(face_bounding_box))

        # PUBLISH THE BOUNDING BOX
        face_box_pub.publish(face_bounding_box)

        # Test for checking box is correct
        # for (x, y, w, h) in face_bounding_box:
        #     cv.rectangle(converted_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # camera_functions.show_image("face_detection node", converted_frame)

#################################################################################
# Uses the Haar feature based cascade classifiers in OpenCV to detect faces
def face_detect(img):

    # Get the current image frame and make a greyscale copy
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Adjust the resolution percentage decrease - FOR FASTER DETECTION
    percentage_decrease = 50

    # Get the width and height dimensions
    width = int(img_gray.shape[1] * percentage_decrease / 100)
    height = int(img_gray.shape[0] * percentage_decrease / 100)
    dim = (width, height)

    # Resize the image to smaller resolution
    img_gray = cv.resize(img_gray, dim, interpolation=cv.INTER_AREA)

    # Detect faces in the image as a bounding box
    faces = faceCascade.detectMultiScale(
        img_gray,
        scaleFactor=1.12,           # Attempting to tune this was originally 1.1
        minNeighbors=5,             # Can adjust to fix latency also, originally was 5
        minSize=(30, 30),
        flags = cv.CASCADE_SCALE_IMAGE
    )

    # Test if face was detected in the frame
    if faces is not tuple():

        # Convert box coordiantes back to original frame resolution
        faces = faces * (100 / percentage_decrease)
        faces = np.array(faces, int)

    # Return the face bounding box
    return faces

#################################################################################
def recognise_face():

    # Initilaise the node and display message 
    rospy.init_node('facial_recognition', anonymous=True)
    rospy.loginfo('facial_recognition\tNODE INIT')
    neural_network()
    # Set the ROS spin rate: 1Hz ~ 1 second
    rate = rospy.Rate(1)        ############ Can make this an argument in launch and streamline rates##############

    # Subscriber callbacks
    rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, camera_callback, queue_size=1)

    # Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
    while not rospy.is_shutdown():
        rospy.spin()

################################################################################
def neural_network():

    ################################ Define some custom transforms to apply to the image ####################
    custom_transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



    # Training data path
    trainingPath = '/home/aidanstapleton/ur5espace/src/exp_mp/scripts/faces'
    trainingSet = []
    print(trainingPath)
    # Get the training images
    #trainset = [cv.imread(file) for file in glob.glob('faces/*.jpg')]

    # Append the training images (selfies) to the training set data list
    for image in trainingPath:

        trainingSet.append(image)

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
    testingSet = trainingSet[0]

    ################################ Dataloaders ###########################################################
    trainloader = torch.utils.data.DataLoader(trainingSet, 
                                            batch_size=8,
                                            shuffle=True)

    testloader = torch.utils.data.DataLoader(testingSet,
                                            batch_size=4,
                                            shuffle=False)

    ################################ Understanding the dataset #############################################
    ################################ This section just prints info #########################################
    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 
    #            'dog', 'frog', 'horse', 'ship', 'truck')
    classes = ('Aidan', 'Antony', 'Kieran', 'Noah')

    # print number of samples                                                       
    print("Number of training samples is {}".format(len(trainingSet)))
    print("Number of test samples is {}".format(len(testingSet)))

    # iterate through the training set print useful information
    dataiter = iter(trainloader)
    images, labels = dataiter.next()    # this gather one batch of data

    print("Batch size is {}".format(len(images)))
    print("Size of each image is {}".format(images[0].shape))

    print("The labels in this batch are: {}".format(labels))
    print("These correspond to the classes: {}, {}, {}, {}".format(
        classes[labels[0]], classes[labels[1]],
        classes[labels[2]], classes[labels[3]]))

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
            self.output_size = 10   # 10 classes

            super(Network, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)             # 2D convolution
            self.pool = nn.MaxPool2d(2, 2)              # max pooling
            self.conv2 = nn.Conv2d(6, 16, 5)            # 2D convolution
            self.fc1 = nn.Linear(16 * 5 * 5, 120)       # Fully connected layer
            self.fc2 = nn.Linear(120, 84)               # Fully connected layer
            self.fc3 = nn.Linear(84, self.output_size)  # Fully connected layer

        def forward(self, x):
            
            # Define the forward pass
            x = self.pool(functional.relu(self.conv1(x)))
            x = self.pool(functional.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
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

        # Calculate loss
        loss = criterion(predicted_labels, labels)
        
        # Perform back propagation and compute gradients
        loss.backward()
        
        # Take a step and update the parameters of the network
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('Epoch: %d, %5d loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    print('Finished Training.')
    torch.save(net.state_dict(), "data/cifar_trained.pth")
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
    try:
        recognise_face()
    except rospy.ROSInterruptException:
        pass
