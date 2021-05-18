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
import glob
import imageio
import time
import io
from shutil import copyfile
import gdown
from IPython import display
from zipfile import ZipFile
from tqdm.notebook import tqdm

def data_pipeline_load(dataset_name, **kwargs):
    """ Create optimized data loading pipeline for training.
    Adapted from: https://cs230.stanford.edu/blog/datapipeline/,
    https://www.tensorflow.org/tutorials/load_data/images#standardize_the_data
    
    Parameters
    ----------
    dataset_name : str
        Name of dataset, either mnist or faces.
    kwargs : dict
        Must include resize_to for both mnist and faces. 
        Must include data_directory for faces.
        If resize_to is None, the images keep their original shapes.
        Must include reduce_to (reduce dataset size). If None, all of the
        dataset is used.
    
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
    Dataset is not pre-processed, shuffled, not batched. 
    and not pre-fetched yet.
    """
    resize_to = kwargs["resize_to"]
    normalizer = kwargs["normalizer"]
    reduce_to = kwargs["reduce_to"]
    
    # load dataset. dataset is shuffled
    if dataset_name == "mnist":
        dataset, dataset_size, img_original_shape = \
          data_pipeline_load_mnist(reduce_to)
    elif dataset_name == "faces":
        data_directory= kwargs["data_directory"]
        dataset, dataset_size, img_original_shape = \
          data_pipeline_load_faces(data_directory, reduce_to)
    
    # if resize_to is None, keep original image shape
    resize_to = img_original_shape if resize_to is None else resize_to
    # pre-process dataset
    dataset = data_pipeline_pre_process(dataset, normalizer, resize_to)
    print(f"loaded and pre-processed: {dataset_name}, size: {dataset_size}")
    
    return dataset, dataset_size


def data_pipeline_pre_process(dataset, normalizer, resize_to):
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
    Dataset is pre-processed - standardized to -1.0 - 1.0 (for tanh) 
    and resized, or to 0.0-1.0 (sigmoid)
    Dataset is shuffled, but not batched and not pre-fetched yet.
    """
    # automatically finds good allocation of CPU budget
    dataset = dataset.map(normalizer, 
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
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
    print(f"dataset: {len(dataset)}")
    # dont shuffe here as it makes training extremely slow
    # do batching and pre-fetch here, dataset is already shuffled in load
    # functions
    dataset = dataset.batch(batch_size)
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


def data_pipeline_load_mnist(reduce_to=None):
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
    reduce_to : int or None
      Reduce the dataset to this many images. If None, the original dataset
      size is used.
    
    Raises
    ------
    None
    
    Notes
    -----
    Dataset is shuffled with buffer size = dataset size (full random shuffle).
    """
    (x_train, y_train), (_, _) = mnist.load_data()
    
    img_original_shape = (28, 28)
    x_train = x_train.reshape(x_train.shape[0], 
                              img_original_shape[0], 
                              img_original_shape[1], 1).astype('float32')
    
    # if needed, reduce dataset size
    if reduce_to is not None:
      x_train = x_train[:reduce_to]
    
    #make dataset
    dataset = tf.data.Dataset.from_tensor_slices(x_train)
    
    # shuffle here since more difficult later
    dataset_size = len(x_train)
    dataset = dataset.shuffle(dataset_size)
    
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
    image = tf.image.decode_jpeg(image_string, channels=3)
    
    #This will convert to float values in [0, 1], so do not use it here.
    # we normalize later.
    #image = tf.image.convert_image_dtype(image, tf.float32)

    return image

def data_pipeline_load_faces(data_directory, reduce_to=None):
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
    Dataset is shuffled with buffer size = dataset size (full random shuffle).
    """
    # get paths to images
    filenames = get_faces_paths(data_directory)
    
    # reduce dataset size if needed
    if reduce_to is not None:
      filenames = filenames[:reduce_to]
      
    dataset_size = len(filenames)
    
    # get original image shape
    img = cv2.imread(filenames[0])
    img_original_shape = img.shape[:2]
    
    # make dataset
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    # Shuffle dataset here (paths, not images, helps RAM)
    dataset = dataset.shuffle(dataset_size)
    # parse paths to images
    dataset = dataset.map(parse_function, 
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    return dataset, dataset_size, img_original_shape


def image_normalizer(typ="tanh"):
    """ Normalize images. Either for sigmoid or tanh activation.
    sigmoid is x/255.0 to be bewteen 0.0-1.0.
    tanh is (x-127.5)/127.5 to be between -1.0-1.0.
    
    Parameters
    ----------
    typ : str
        Type of activation function to normalize for.
    
    Returns
    -------
    normalizer : tf.Rescaling
      The image normalizer. Will be mapped to each image in dataset.
    
    Raises
    ------
    None
    
    Notes
    -----
    None
    """
    assert typ in ["tanh", "sigmoid"], f"invalid image normalizer"
    if typ == "tanh":
      normalizer = \
        tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, 
                                                             offset=-1)
    else:
      normalizer = \
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255.0)

    return normalizer


def image_rescaler(typ="tanh"):
    """ Inverse Normalize (rescale) images. 
    Either for sigmoid or tanh activation.
    sigmoid is x/255.0 to be bewteen 0.0-1.0.
    tanh is (x-127.5)/127.5 to be between -1.0-1.0.
    
    Parameters
    ----------
    typ : str
        Type of activation function to normalize for.
    
    Returns
    -------
    rescaler : tf.Rescaling
      The image rescaler. Will be mapped to each image for visualization
      purposes.
    
    Raises
    ------
    None
    
    Notes
    -----
    None
    """
    assert typ in ["tanh", "sigmoid"], f"invalid image rescaler"
    if typ == "tanh":
      rescaler = lambda image: (image.numpy() * 127.5 + 127.5).astype("int32")
    else:
      rescaler = lambda image: (image.numpy() * 255.0).astype("int32")

    return rescaler