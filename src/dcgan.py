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
    '''
        Guidelines from https://arxiv.org/pdf/1511.06434.pdf
            • Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).
            • Use batchnorm in both the generator and the discriminator.
            • Remove fully connected hidden layers for deeper architectures.
            • Use ReLU activation in generator for all layers except for the output, which uses Tanh.
            • Use LeakyReLU activation in the discriminator for all layers.

    '''

    def __init__(self, input_dim, num_layers_gen, num_layers_disc, name, batchNorm=True):
        super(DCGAN, self).__init__(name=name)
        self.isBatch = batchNorm

        # This is probaly the best place to call both generator and discriminator
        gen = self.generator(input_dim, num_layers_gen)

    def generator(self, input_dim, num_layers_gen):
        '''
        Generator that allows us to define the number of layers and number of filters according to the data at hand

        Parameters
        ----------

        input_dim : int
            The number of input dimensions, that is, the number of nodes in the first layer of the encoder and the
            last, output layer of the decoder.

        num_layers_gen : int
            The number of layers for the generator model. This will allow the user to easily adapt the network architecture to the
            data at hand.

        num_layers_disc : int
            The number of layers for the discriminator model. This will allow the user to easily adapt the network architecture to the
            data at hand.

        name : str
            The name of the model.
        '''
        gen_model = keras.Sequential()
        # Didn't have time to test this (!!!!) <- Number of filter is assumed to be half of the previous layer (except the output, which should be 1)
        for i in range(num_layers_gen-2):
            gen_model.add(keras.layers.Convolution2DTranspose())
            if self.isBatch:
                gen_model.add(keras.layers.BatchNormalization())
            gen_model.add(keras.layers.ReLU())

        # Last layer
        gen_model.add(keras.layers.Convolution2DTranspose())
        if self.isBatch:
            gen_model.add(keras.layers.BatchNormalization())
            gen_model.add(keras.layers.ReLU())
        gen_model.add(keras.layers.Activation('tanh'))

        return gen_model

    def discriminator(self):
        disc = keras.Sequential()
        # deconv2d
        # leaky relu
        # deconv2d
        # leaky relu
        # deconv2d
        # leaky relu
        # deconv2d
        # leaky relu
    '''
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
