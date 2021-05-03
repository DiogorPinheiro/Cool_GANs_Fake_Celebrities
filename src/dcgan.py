import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import yaml
from datetime import datetime
import cv2      # Need to install OpenCV (python3 -m pip install opencv-python)
import os

from tensorflow.keras.datasets import mnist

# print(tf.__version__)   # Check your TF version (should be 2.4.0)

DATA_DIRECTORY = '../data/celebA/images'
'''
class DCGAN(object):
    def __init__(self, input_dim, name):
    
    def generator(self):
    # conv2d
    # leaky relu
    # conv2d
    # leaky relu
    # conv2d
    # leaky relu
    # conv2d
    # leaky relu
    # conv2d
    # leaky relu
    # sigmoid
    
    def discriminator(self):
    # relu
    # deconv2d
    # deconv2d
    # deconv2d
    # deconv2d
    # tanh
    
    def train(self):
    
    def save(self):
    
    def load(self):

'''

if __name__ == "__main__":
    # Contributed: Diogo Pinheiro, Jakob Lindén, Márk Csizmadia, Patrick Jonsson

    data = 'MNIST'      # options : 'MNIST' or 'Faces'

    if data == 'MNIST':
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()    # load MNIST data

        # Normalize data to [-1,1] because we'll use tanh activation function
        X_train /= 255      # [0,1]
        X_train = (X_train-0.5)/0.5  # [-1,1]
    elif data == 'FACES':
        faces_data = []
        for filename in os.listdir(DATA_DIRECTORY):
            img = cv2.imread(os.path.join(DATA_DIRECTORY, filename))
            if img is not None:
                faces_data.append(img)
