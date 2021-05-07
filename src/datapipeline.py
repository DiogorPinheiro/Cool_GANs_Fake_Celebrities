from tensorflow.keras.datasets import mnist
import cv2
from datetime import datetime
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
import os
from PIL import Image
import math
import yaml
from copy import deepcopy

def data_pipeline_load(dataset_name, **kwargs):
    """ Create optimized data loading pipeline for training.
    Adapted from: https://cs230.stanford.edu/blog/datapipeline/,
    https://www.tensorflow.org/tutorials/load_data/images#standardize_the_data
    
    Parameters
    ----------
    dataset_name : str
        Name of dataset, either mnist or faces.
    kwargs : dict
        Must include resize_to for both nist and faces. Must include data_directory for faces.
        If resize_to is None, the images keep their original shapes.
    
    Returns
    -------
    dataset : tf.data.Dataset
        Dataset object.
    dataset_size : int
        Number of instances in dataset. For shuffling later.
    
    Raises
    ------
    None
    
    Notes
    -----
    Dataset is not pre-processed, not shuffled, not batched and not pre-fetched yet.
    """
    resize_to = kwargs["resize_to"]
    
    if dataset_name == "mnist":
        dataset, dataset_size, img_original_shape = data_pipeline_load_mnist()
    elif dataset_name == "faces":
        data_directory= kwargs["data_directory"]
        dataset, dataset_size, img_original_shape = data_pipeline_load_faces(data_directory)
    
    resize_to = img_original_shape if resize_to is None else resize_to
    
    dataset = data_pipeline_pre_process(dataset, resize_to)
    
    print(f"loaded and pre-processed: {dataset_name}, size: {dataset_size}")
    
    return dataset, dataset_size

def data_pipeline_pre_process(dataset, resize_to):
    """ Create optimized data pre-processing pipeline for training.
    Adapted from: https://cs230.stanford.edu/blog/datapipeline/,
    https://www.tensorflow.org/tutorials/load_data/images#standardize_the_data
    
    Parameters
    ----------
    dataset_name : str
        Name of dataset, either mnist or faces.
    resize_to : tuple
        Resize to (height, width).
    
    Returns
    -------
    dataset : tf.data.Dataset
        Dataset object.
    
    Raises
    ------
    None
    
    Notes
    -----
    Dataset is pre-processed - standardized to -1.0 - 1.0 and resized.
    But not shuffled, not batched and not pre-fetched yet.
    """
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset=-1)
    
    # automatically finds good allocation of CPU budget
    dataset = dataset.map(normalization_layer, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    
    # automatically finds good allocation of CPU budget
    dataset = dataset.map(lambda im: resize_image(im, resize_to), 
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    return dataset

def data_pipeline_pre_train(dataset, dataset_size, batch_size):
    """ Create optimized pre-training data pipeline. Shuffle, batch, and pre-fetch dataset.
    Adapted from: https://cs230.stanford.edu/blog/datapipeline/,
    https://www.tensorflow.org/tutorials/load_data/images#standardize_the_data
    
    Parameters
    ----------
    dataset : tf.data.Dataset
        Dataset object.
    dataset_size : int
        Number of instances in dataset. For shuffling.
    batch_size : int 
        Batch size.
    
    Returns
    -------
    dataset : tensorflow.data.Dataset
        Dataset object.
    
    Raises
    ------
    None
    
    Notes
    -----
    Dataset is now pre-processed, shuffled, batched and pre-fetched yet.
    buffer_size (dataset_size) >= batch_size to get an uniform shuffle, right? idk
    """
    dataset = dataset.shuffle(dataset_size).batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset


def resize_image(image, resize_to):
    """ Resize image.
    
    Parameters
    ----------
    image : tensorflow.Tensor
        Image object.
    resize_to : tuple
        Resize to (height, width).
    
    Returns
    -------
    resized_image : tensorflow.Tensor
        Resize image object.
    
    Raises
    ------
    None
    
    Notes
    -----
    None
    """
    resized_image = tf.image.resize(image, [resize_to[0], resize_to[1]])
    
    return resized_image


def data_pipeline_load_mnist():
    """ MNIST loading pipeline.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    dataset : tf.data.Dataset
        Dataset object.
    dataset_size : int
        Number of instances in dataset. For shuffling.
    
    Raises
    ------
    None
    
    Notes
    -----
    None
    """
    (x_train, y_train), (_, _) = mnist.load_data()
    
    img_original_shape = (28, 28)
    
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    
    dataset = tf.data.Dataset.from_tensor_slices(x_train)
    
    # shuffle here since more difficult later
    dataset_size = len(x_train)
    
    return dataset, dataset_size, img_original_shape

def get_faces_paths(data_directory):
    """ Get the paths to the faces images.
    
    Parameters
    ----------
    data_directory : str
        Path to the faces images.
    
    Returns
    -------
    paths_faces : list
        List of paths to all images.
    
    Raises
    ------
    None
    
    Notes
    -----
    None
    """
    paths_faces = []
        
    for idx, filename in enumerate(os.listdir(data_directory)):
        paths_face = os.path.join(data_directory, filename)
        paths_faces.append(paths_face)
    
    return paths_faces

def parse_function(filename):
    """ Parse a filename and load the corresponding image. Used for faces.
    
    Parameters
    ----------
    filename : str
        Path to the faces image.
    
    Returns
    -------
    image : tensorflow.Tensor
        Image object.
    
    Raises
    ------
    None
    
    Notes
    -----
    None
    """
    image_string = tf.io.read_file(filename)

    #Don't use tf.image.decode_image, or the output shape will be undefined
    # automatically finds good allocation of CPU budget
    image = tf.image.decode_jpeg(image_string, channels=3)
    
    #This will convert to float values in [0, 1]
    #image = tf.image.convert_image_dtype(image, tf.float32)

    return image

def data_pipeline_load_faces(data_directory):
    """ Faces loading pipeline.
    
    Parameters
    ----------
    data_directory : str
        Path to the faces images.
    
    Returns
    -------
    dataset : tf.data.Dataset
        Dataset object.
    dataset_size : int
        Number of instances in dataset. For shuffling.
    
    Raises
    ------
    None
    
    Notes
    -----
    None
    """
    filenames = get_faces_paths(data_directory)
    dataset_size = len(filenames)
    
    img = cv2.imread(filenames[0])
    img_original_shape = img.shape[:2]
    
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    #dataset = dataset.shuffle(len(filenames))
    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    return dataset, dataset_size, img_original_shape