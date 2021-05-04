from tensorflow.keras.datasets import mnist
import cv2      # Need to install OpenCV (python3 -m pip install opencv-python)
from datetime import datetime
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import os
os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'  # minimize TF annoying messages


# print(tf.__version__)   # Check your TF version (should be 2.4.0)

DATA_DIRECTORY = '../data/celebA/images'


class DCGAN(keras.Model):
    def __init__(self, input_dim, name):
        super(DCGAN, self).__init__(name=name)

        self.generator(input_dim)

    def generator(self, input_dim):
        # take care of input z (!!)

        # Functional API for more flexibility
        inputs = keras.Input(shape=input_dim)
        # NOTE: Number of filters should be adapted to each dataset <- FIND WAY TO DO IT AUTOMATICALLY
        genx = keras.layers.Convolution2DTranspose()(inputs)
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
    '''
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


def data_pipeline(images, batch_size):
    '''
        Create optimized data pipeline for training
            NOTE: Ideally this should work for both datasets <- have to check 

        Adapted from : https://cs230.stanford.edu/blog/datapipeline/

        buffer_size >= data_size to get an uniform shuffle
    '''
    data = tf.data.Dataset.from_tensor_slices(images)
    data = data.shuffle(len(images)).batch(batch_size)
    data = data.map(
        images, num_parallel_calls=tf.data.experimental.AUTOTUNE)   # automatically finds good allocation of CPU budget

    return data.prefetch(tf.data.experimental.AUTOTUNE)


if __name__ == "__main__":
    # Contributed: Diogo Pinheiro, Jakob Lindén, Márk Csizmadia, Patrick Jonsson

    data = 'MNIST'      # options : 'MNIST' or 'Faces'

    if data == 'MNIST':
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()    # load MNIST data
        print("MNIST Data Shape -> {}".format(X_train.shape))

        # Normalize data to [-1,1] because we'll use tanh activation function
        X_train /= 255      # [0,1]
        X_train = (X_train-0.5)/0.5  # [-1,1]

        batch_size = 100    # FIND BEST VALUE !!!!

    # Should be resized to 32*32 images, although we can experiment with smaller sizes (faster training)
    elif data == 'FACES':
        faces_data = []
        for filename in os.listdir(DATA_DIRECTORY):
            img = cv2.imread(os.path.join(DATA_DIRECTORY, filename))
            if img is not None:
                faces_data.append(img)

    trainX = data_pipeline(X_train, batch_size)
