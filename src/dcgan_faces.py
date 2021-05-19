import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import yaml
import os
import math
import yaml
import imageio
import time
import io
import gdown
import numpy
from numpy.random import randint
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize
from tensorflow.keras.datasets import mnist
from datetime import datetime
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
from copy import deepcopy
from shutil import copyfile
from IPython import display
from zipfile import ZipFile
from random import shuffle
from glob import glob
from shutil import copy
from datapipeline import *
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import randint
from scipy.linalg import sqrtm
from tqdm import tqdm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize

# To install tensorflow_docs pip install -q git+https://github.com/tensorflow/docs


os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'  # minimize TF annoying messages

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Remove GPU 


# Make tf work on 3060 GPU

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

tf.config.run_functions_eagerly(True)


print(tf.__version__)   # Check your TF version (should be 2.4.0)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

DATA_DIRECTORY = '../data/celebA/images'

print(tf.__version__)
print(f"GPUs used: {tf.config.list_physical_devices('GPU')}")

def make_generator_1_faces(input_shape, output_shape):
    """ Make the generator model.

    Parameters
    ----------
    input_shape : tuple
        Input latent vector shape: (latent_dim, )
    output_shape : tuple
        Output image shape: (height, width, channels)

    Returns
    -------
    keras.Model
        The generator model.

    Raises
    ------
    None

    Notes
    -----
    code based on: https://www.tensorflow.org/tutorials/generative/dcgan
    model: the paper
    pair: make_discriminator_1_faces
    note: output_shape = (64,64,3)
    """
    inputs = keras.Input(shape=input_shape, name="generator_input")

    z_proj = layers.Dense(4*4*1024, use_bias=False, name="g_h0_lin")(inputs)
    h = layers.BatchNormalization(name="g_h0_lin_bn")(z_proj)
    h = layers.LeakyReLU(name="g_h0_lin_a")(h)
    h = layers.Reshape((4, 4, 1024), name="g_h0_a")(h)
    # Note: None is the batch size
    assert tuple(h.shape) == (None, 4, 4, 1024)

    h = layers.Conv2DTranspose(filters=512, kernel_size=(5, 5), 
                                     strides=(2, 2), padding='same', 
                                     use_bias=False, name="g_h1")(h)
    h = layers.BatchNormalization(name="g_h1_bn")(h)
    h = layers.LeakyReLU(name="g_h1_a")(h)
    assert tuple(h.shape) == (None, 8, 8, 512)

    h = layers.Conv2DTranspose(filters=256, kernel_size=(5, 5), 
                                     strides=(2, 2), padding='same', 
                                     use_bias=False, name="g_h2")(h)
    h = layers.BatchNormalization(name="g_h2_bn")(h)
    h = layers.LeakyReLU(name="g_h2_a")(h)
    assert tuple(h.shape) == (None, 16, 16, 256)



    h = layers.Conv2DTranspose(filters=128, kernel_size=(5, 5), 
                                     strides=(2, 2), padding='same', 
                                     use_bias=False, name="g_h3")(h)
    h = layers.BatchNormalization(name="g_h3_bn")(h)
    h = layers.LeakyReLU(name="g_h3_a")(h)
    assert tuple(h.shape) == (None, 32, 32, 128)

    h = layers.Conv2DTranspose(filters=3, kernel_size=(5, 5), 
                               strides=(2, 2), padding='same', 
                               use_bias=False, name="g_h4_a",
                               activation='tanh')(h)
    assert tuple(h.shape) == (None, 64, 64, 3)

    generator = tf.keras.Model(inputs=inputs, outputs=h, name="generator")

    return generator


def make_discriminator_1_faces(output_shape):
    """ Make the discriminator model.

    Parameters
    ----------
    output_shape : tuple
        Output image shape, height, width, channels.

    Returns
    -------
    keras.Model
        The discriminator model.

    Raises
    ------
    None

    Notes
    -----
    code based on: https://www.tensorflow.org/tutorials/generative/dcgan
    model: the paper does not mention exact discriminator architecture,
    so trying to mirror the generator pair
    pair: make_generator_1_faces
    note: output_shape = (64,64,3)
    """
    outputs = keras.Input(shape=output_shape, name="discriminator_input")
    
    h = layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(2, 2), 
                      padding='same', name="d_h0")(outputs)
    h = layers.LeakyReLU(alpha=0.3)(h)
    h = layers.Dropout(rate=0.3)(h)

    h = layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(2, 2), 
                      padding='same', name="d_h1")(h)
    h = layers.LeakyReLU(alpha=0.3)(h)
    h = layers.Dropout(rate=0.3)(h)

    h = layers.Conv2D(filters=512, kernel_size=(5, 5), strides=(2, 2), 
                      padding='same', name="d_h2")(h)
    h = layers.LeakyReLU(alpha=0.3)(h)
    h = layers.Dropout(rate=0.3)(h)

    h = layers.Conv2D(filters=1024, kernel_size=(5, 5), strides=(2, 2), 
                      padding='same', name="d_h3")(h)
    h = layers.LeakyReLU(alpha=0.3)(h)
    h = layers.Dropout(rate=0.3)(h)

    h = layers.Flatten(name="d_h4_lin")(h)
    h = layers.Dense(units=1)(h)
    
    discriminator = tf.keras.Model(inputs=outputs, outputs=h, 
                                   name="discriminator")

    return discriminator


def make_generator_2_faces(input_shape, output_shape):
    """ Make the generator model.

    Parameters
    ----------
    input_shape : tuple
        Input latent vector shape: (latent_dim, )
    output_shape : tuple
        Output image shape: (height, width, channels)

    Returns
    -------
    keras.Model
        The generator model.

    Raises
    ------
    None

    Notes
    -----
    code based on: https://www.tensorflow.org/tutorials/generative/dcgan
    model: the paper, but strides=(1,1) instead of (2,2) in g_h4_a to get
    32x32 output shape (smaller resolution faces)
    pair: make_discriminator_2_faces
    note: output_shape = (32,32,3)
    """
    inputs = keras.Input(shape=input_shape, name="generator_input")

    z_proj = layers.Dense(4*4*256, use_bias=False, name="g_h0_lin")(inputs)
    h = layers.BatchNormalization(name="g_h0_lin_bn")(z_proj)
    h = layers.LeakyReLU(name="g_h0_lin_a")(h)
    h = layers.Reshape((4, 4, 256), name="g_h0_a")(h)
    # Note: None is the batch size
    assert tuple(h.shape) == (None, 4, 4, 256)

    h = layers.Conv2DTranspose(filters=256, kernel_size=(5, 5), 
                                     strides=(2, 2), padding='same', 
                                     use_bias=False, name="g_h1")(h)
    h = layers.BatchNormalization(name="g_h1_bn")(h)
    h = layers.LeakyReLU(name="g_h1_a")(h)
    assert tuple(h.shape) == (None, 8, 8, 256)

    h = layers.Conv2DTranspose(filters=128, kernel_size=(5, 5), 
                                     strides=(2, 2), padding='same', 
                                     use_bias=False, name="g_h2")(h)
    h = layers.BatchNormalization(name="g_h2_bn")(h)
    h = layers.LeakyReLU(name="g_h2_a")(h)
    assert tuple(h.shape) == (None, 16, 16, 128)



    h = layers.Conv2DTranspose(filters=64, kernel_size=(5, 5), 
                                     strides=(2, 2), padding='same', 
                                     use_bias=False, name="g_h3")(h)
    h = layers.BatchNormalization(name="g_h3_bn")(h)
    h = layers.LeakyReLU(name="g_h3_a")(h)
    assert tuple(h.shape) == (None, 32, 32, 64)

    h = layers.Conv2DTranspose(filters=3, kernel_size=(5, 5), 
                               strides=(1, 1), padding='same', 
                               use_bias=False, name="g_h4_a",
                               activation='tanh')(h)
    assert tuple(h.shape) == (None, 32, 32, 3), f"shape={tuple(h.shape)}"

    generator = tf.keras.Model(inputs=inputs, outputs=h, name="generator")

    return generator


def make_discriminator_2_faces(output_shape):
    """ Make the discriminator model.

    Parameters
    ----------
    output_shape : tuple
        Output image shape, height, width, channels.

    Returns
    -------
    keras.Model
        The discriminator model.

    Raises
    ------
    None

    Notes
    -----
    code based on: https://www.tensorflow.org/tutorials/generative/dcgan
    the paper does not mention exact discriminator architecture,
    so trying to mirror the generator pair
    pair: make_generator_2_faces
    note: output_shape = (32,32,3)
    """
    outputs = keras.Input(shape=output_shape, name="discriminator_input")
    
    h = layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), 
                      padding='same', name="d_h0")(outputs)
    h = layers.LeakyReLU(alpha=0.3)(h)
    h = layers.Dropout(rate=0.3)(h)

    h = layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(2, 2), 
                      padding='same', name="d_h1")(h)
    h = layers.LeakyReLU(alpha=0.3)(h)
    h = layers.Dropout(rate=0.3)(h)

    h = layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(2, 2), 
                      padding='same', name="d_h2")(h)
    h = layers.LeakyReLU(alpha=0.3)(h)
    h = layers.Dropout(rate=0.3)(h)

    h = layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(2, 2), 
                      padding='same', name="d_h3")(h)
    h = layers.LeakyReLU(alpha=0.3)(h)
    h = layers.Dropout(rate=0.3)(h)

    h = layers.Flatten(name="d_h4_lin")(h)
    h = layers.Dense(units=1)(h)
    
    discriminator = tf.keras.Model(inputs=outputs, outputs=h, 
                                   name="discriminator")

    return discriminator
    
def discriminator_loss(real_output, fake_output, cross_entropy):
    """ Compute the discriminator loss.

    Parameters
    ----------
    real_output : tf.Tensor
        Real image batch of shape (None, height, width, channels).
    fake_output : tf.Tensor
        Fake image batch of shape (None, height, width, channels).

    Returns
    -------
    total_loss : tf.Tensor
        Discriminator loss. A combination of how well the discriminator can tell
        if an image is fake and if it is real.

    Raises
    ------
    None

    Notes
    -----
    None
    """
    # real_loss: binary cross-entropy for if the discrimnator predicts 1 for a
    # real image
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    
    # fake_loss: binary cross-entropy for if the discrimnator predicts 0 for a
    # fake image
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    
    # total_loss: want all predictions for reals to be 1 (low real_loss) and
    # 0 for fakes (low fake_loss)
    total_loss = real_loss + fake_loss

    return total_loss

def generator_loss(fake_output, cross_entropy):
    """ Compute the generator loss.

    Parameters
    ----------
    fake_output : tf.Tensor
        Fake image batch of shape (None, height, width, channels).

    Returns
    -------
    tf.Tensor
        Generator loss. A measure of how well the generator decieves the 
        discriminator.

    Raises
    ------
    None

    Notes
    -----
    None
    """
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step_old(images, batch_size, latent_dim, generator, discriminator,
               generator_loss, discriminator_loss, cross_entropy,
               generator_optimizer, discriminator_optimizer):
    #noise = tf.random.normal([batch_size, latent_dim])
    random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(random_latent_vectors, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output, cross_entropy)
      disc_loss = discriminator_loss(real_output, fake_output, cross_entropy)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled" (speeds up training).
@tf.function
def train_step(images, batch_size, latent_dim, generator, discriminator,
               cross_entropy, generator_optimizer, discriminator_optimizer):
    """ Train step. Exectued for every batch.

    Parameters
    ----------
    images : tf.Tensor
        Image batch of shape (batch_size, height, width, channels)
    batch_size : int
      The batch size (has to be same to what is used to batch the dataset in
      the data pipeline).
    latent_dim : int
      The latent vector dimension of the generator.
    generator : keras.Model
      The generator model.
    discriminator : keras.Model
      The discriminator model.
    cross_entropy : func
      Helper fucntion for binary cross-entropy so we do not have to always 
      instantiate one (RAM friendly, doesnt matter too much tho).
    generator_optimizer : keras.optimizers
      The generator optimizer.
    discriminator_optimizer : keras.optimizers
      The discriminator optimizer.

    Returns
    -------
    gen_loss : tf.Tensor
      The generator loss.
    disc_loss : tf.Tensor
      The discriminator loss.

    Raises
    ------
    None

    Notes
    -----
    None
    """
    # Sample random points in the latent space
    random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))

    # Decode them to fake images
    generated_images = generator(random_latent_vectors, training=False)

    # Combine them with real images on axis=0 (i.e.: now have the generated
    # and the real images on one "line")
    combined_images = tf.concat([generated_images, images], axis=0)

    # Assemble labels discriminating real from fake images.
    labels = tf.concat(
        [tf.ones((batch_size, 1)), tf.zeros((tf.shape(images)[0], 1))], axis=0
    )

    # Add random noise to the labels - important trick according to F.C.
    # https://keras.io/examples/generative/dcgan_overriding_train_step/
    labels += 0.05 * tf.random.uniform(tf.shape(labels))

    # Train the discriminator.
    with tf.GradientTape() as tape:
        predictions = discriminator(combined_images, training=True)
        disc_loss = cross_entropy(labels, predictions)
    
    # get and apply grads for discriminator
    grads = tape.gradient(disc_loss, discriminator.trainable_weights)
    discriminator_optimizer.apply_gradients(
        zip(grads, discriminator.trainable_weights)
    )

    # Sample random points in the latent space
    random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))

    # Assemble labels that say "all real images"
    misleading_labels = tf.zeros((batch_size, 1))

    # Train the generator (note that we should *not* update the weights
    # of the discriminator)!
    with tf.GradientTape() as tape:
        predictions = discriminator(generator(random_latent_vectors, 
                                              training=True), 
                                    training=False)
        gen_loss = cross_entropy(misleading_labels, predictions)
    
    # get and apply grads for generator
    grads = tape.gradient(gen_loss, generator.trainable_weights)
    generator_optimizer.apply_gradients(zip(grads, generator.trainable_weights))

    return gen_loss, disc_loss  

def fid_step(batch_size, latent_dim, generator, images, fid_scorer):
    """ FID step. Exectued for every batch in the FID test set if wanted.

    Parameters
    ----------
    batch_size : int
      The batch size.
    latent_dim : int
      The latent vector dimension of the generator.
    generator : keras.Model
      The generator model.
    images : tf.Tensor
      Image batch of shape (batch_size, width, height, channels).
      Note that the image batch is normalized, so the FID scorer takes care of 
      rescaling them to 0-255 before feeding them into the Inception model.
    fid_scorer : FID
      FID score computer.

    Returns
    -------
    None

    Raises
    ------
    None

    Notes
    -----
    Note that the image batch is normalized, so the FID scorer takes care of 
    rescaling them to 0-255 before feeding them into the Inception model.
    Also note that the FID can only be computed for images with 3 channels.
    """
    # Sample random points in the latent space
    random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))

    # Decode them to fake images
    generated_images = generator(random_latent_vectors, training=False)
    
    # convert tf.Tensor image batch to np.ndarray
    generated_images_np = generated_images.numpy()
    images_np = images.numpy()
    
    # compute the fid score (apply_rescaler has to be true if the images
    # are pre-processed with a normalizer)
    fid_score = fid_scorer(generated_images_np, images_np, apply_rescaler=True)

    return fid_score

def train(dataset, gen_loss_metric, disc_loss_metric,
          train_summary_writer, gen_summary_writer,
          epochs, batch_size, latent_dim, generator, discriminator, 
          cross_entropy, generator_optimizer, discriminator_optimizer, 
          seed, viz_save_path, checkpoint_prefix, dataset_name, dataset_size,
          rescaler, ckpt_save_epoch, **fid_dict):
    """ Train loop for training a DCGAN model with a generator and a 
    discriminator.

    Parameters
    ----------
    dataset : tf.Dataset
        The dataset, pre-processed, shuffled, batched, pre-fetched. 
    gen_loss_metric : keras.Metrics
      Generator loss metric for logging and averaging over batches for an 
      epoch estimate.
    disc_loss_metric : keras.Metrics
      Disciminator loss metric for logging and averaging over batches for an 
      epoch estimate.
    train_summary_writer : tf.summary.SummaryWriter
      For training Tensrboard logs.
    gen_summary_writer : tf.summary.SummaryWriter
      For generated images Tensrboard logs.
    epochs : int
      Epochs.
    batch_size : int
      The batch size.
    latent_dim : int
      The latent vector dimension of the generator.
    generator : keras.Model
      The generator model.
    discriminator : keras.Model
      The discriminator model.
    cross_entropy : func
      Helper fucntion for binary cross-entropy so we do not have to always 
      instantiate one (RAM friendly, doesnt matter too much tho).
    generator_optimizer : keras.optimizers
      The generator optimizer.
    discriminator_optimizer : keras.optimizers
      The discriminator optimizer.
    seed : tf.Tensor
      Random seed for generating and inspecting progress on the same images.
      I.e.: same face or digit over epochs in visualization.
    viz_save_path : str
      Save path for generated images.
    checkpoint_prefix : str
      For checkpoints.
    dataset_name : str
      The dataset name. Herer for visualization reasons.
    dataset_size : int
      The datast size, here for tqdm stuff.
    rescaler : tf.keras.Preprocessing
      Invese of normlaizer for images, here for visualization reasons.
    ckpt_save_epoch : int
      Checkpoint save every n epochs.
    fid_dict : dict
      Contains stuff for computing FID if needed, such as:
      dataset_fid : tf.Dataset, Same as dataset, but only for FID.
      fid_size : int, Number of images for FID computation.
      fid_score_metric : keras.Metrics, Disciminator loss metric for logging 
      and averaging over batches for an epoch estimate.
      fid_scorer : FID, fid scorer object, has image rescaler with the inverse 
      rule of the image dataset normalizer

    Returns
    -------
    None

    Raises
    ------
    None

    Notes
    -----
    None
    """
    # if fid_dict is not empty
    # retrive here as it is used over epochs
    if fid_dict:
      fid_score_metric = fid_dict["fid_score_metric"]
    
    for epoch in range(epochs):
      print(f"epoch: {epoch+1}/{epochs}...")
      start = time.time()

      # tqdm stuff for loading bars
      pbar = tqdm(total=int(dataset_size/batch_size))
      pbar.set_description("batches")

      # mini-batch steps
      for image_batch in dataset:
        gen_loss, disc_loss = \
          train_step(image_batch, batch_size, latent_dim, 
                    generator, discriminator, cross_entropy,
                    generator_optimizer, discriminator_optimizer)
      
        # Update training metrics
        gen_loss_metric.update_state(gen_loss)
        disc_loss_metric.update_state(disc_loss)

        # tqdm render
        pbar.update(1)
        pbar.set_description(f"batches (g_l: {gen_loss_metric.result():.4f}, "
          f"d_l: {disc_loss_metric.result():.4f})")
      
      pbar.close()

      # if fid_dict is not empty
      # do mini-batch-based FID comptation on FID test set
      if fid_dict:
        dataset_fid = fid_dict["dataset_fid"]
        fid_size = fid_dict["fid_size"]
        fid_scorer = fid_dict["fid_scorer"]

        pbar = tqdm(total=int(fid_size/batch_size))
        pbar.set_description("fid batches")
        
        for image_batch in dataset_fid:
          fid_score = fid_step(batch_size, latent_dim, generator, image_batch, 
                              fid_scorer)
          fid_score_metric.update_state(fid_score)
          pbar.update(1)
          pbar.set_description(f"fid batches (fid: {fid_score_metric.result():.4f}")
        
        pbar.close()

      # Save the model every ckpt_save_epoch epochs
      if_ckpt_save_epoch = (epoch + 1) % ckpt_save_epoch == 0
      if if_ckpt_save_epoch:
        checkpoint.save(file_prefix = checkpoint_prefix)

      # Produce images for the GIF as you go
      display.clear_output(wait=True)
      fig = generate_and_save_images(generator,
                              epoch + 1,
                              seed,
                              viz_save_path,
                              dataset_name,
                              rescaler)
      
      # log to tensorboard the generated image grid
      with gen_summary_writer.as_default():
        tf.summary.image("generated_images", plot_to_image(fig), step=epoch)

      # log to tensorboard the training metrics
      with train_summary_writer.as_default():
        tf.summary.scalar('gen_loss_metric', gen_loss_metric.result(), step=epoch)
        tf.summary.scalar('disc_loss_metric', disc_loss_metric.result(), step=epoch)
        if fid_dict:
          tf.summary.scalar('fid_score_metric', fid_score_metric.result(), step=epoch)
      
      # viz images (have to do it here, otherwise can break tensorboard)
      plt.show()

      # show training info
      if fid_dict:
        print(f"{epoch+1}/{epochs} ({time.time()-start:.4} s):\n"
              f"gen_loss_metric={gen_loss_metric.result()},\n"
              f"disc_loss_metric={disc_loss_metric.result()},\n" 
              f"fid_score_metric={fid_score_metric.result()}\n")
      else:
          print(f"{epoch+1}/{epochs} ({time.time()-start:.4} s):\n"
              f"gen_loss_metric={gen_loss_metric.result()},\n"
              f"disc_loss_metric={disc_loss_metric.result()}\n")
      
      # Reset metrics every epoch
      # is this needed? -> yes
      gen_loss_metric.reset_states()
      disc_loss_metric.reset_states()
      if fid_dict:
        fid_score_metric.reset_states()

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                            epochs,
                            seed,
                            viz_save_path,
                            dataset_name,
                            rescaler)
    

def generate_and_save_images(model, epoch, test_input, viz_save_path, dataset_name, rescaler):
    # source: https://www.tensorflow.org/tutorials/generative/dcgan
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.grid(False)
        plt.imshow(rescaler(predictions[i, :, :, 0]), cmap='gray')
        plt.imshow(rescaler(predictions[i]))
        plt.axis('off')

    plt.savefig(os.path.join(viz_save_path, f"epoch_{epoch:04d}.png"))
    #plt.show()

    return fig

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.

    From: https://www.tensorflow.org/tensorboard/image_summaries
    """
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

# scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
      # resize with nearest neighbor interpolation
      new_image = resize(image, new_shape, 0)
      # store
      images_list.append(new_image)
    return asarray(images_list)
 
# calculate frechet inception distance
def calculate_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
      covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

class FID():
  def __init__(self, rescaler):
    # image rescaler
    self.rescaler = rescaler
    # prepare the inception v3 model
    self.input_shape = (299,299,3)
    self.model = InceptionV3(include_top=False, pooling='avg', 
                             input_shape=(299,299,3))

  def scale_images(self, images):
    # scale an array of images to a new size

    images_list = list()
    for image in images:
      # resize with nearest neighbor interpolation
      new_image = resize(image, self.input_shape, 0)
      # store
      images_list.append(new_image)
    
    return asarray(images_list)

  def calculate_fid(self, images1, images2):
    # calculate frechet inception distance

    # calculate activations
    act1 = self.model.predict(images1)
    act2 = self.model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
      covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

  def rescale_images(self, images):
    images_tf = tf.convert_to_tensor(images, np.float32)
    images_tf = tf.map_fn(fn=self.rescaler, elems=images_tf)
    return images_tf.numpy()

  def __call__(self, images1, images2, apply_rescaler):
    # images1 and images2 of shape (batch_size, original_width, original_height, dim)
    # and of type np.ndarray
    assert isinstance(images1, np.ndarray)
    assert isinstance(images2, np.ndarray)

    # convert integer to floating point values
    images1 = images1.astype('float32')
    images2 = images2.astype('float32')
    
    # rescale images fom -1.0-1.0 or 0.0-1.0 to 0.0-255.0 if needed
    if apply_rescaler:
      images1 = self.rescale_images(images1)
      images2 = self.rescale_images(images2)

    # resize images
    images1 = self.scale_images(images1)
    images2 = self.scale_images(images2)
    #print('Scaled', images1.shape, images2.shape)
    
    # pre-process images for Inception
    images1 = preprocess_input(images1)
    images2 = preprocess_input(images2)

    # fid between images1 and images1
    fid = self.calculate_fid(images1, images2)
    #print('FID (in func): %.3f' % fid)

    return fid

if __name__ == "__main__":
    # Contributed: Diogo Pinheiro, Jakob Lindén, Márk Csizmadia, Patrick Jonsson

    """ Download faecs
    os.makedirs("celeba_gan")

    url = "https://drive.google.com/uc?id=1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684"
    output = "celeba_gan/data.zip"
    gdown.download(url, output, quiet=False)

    with ZipFile("celeba_gan/data.zip", "r") as zipobj:
        zipobj.extractall("celeba_gan")
    """
    
    # Train on faces

    # get image normalizer and inverse normlaizer
    image_normalizer_typ = "tanh"
    normalizer_faces = image_normalizer(image_normalizer_typ)
    rescaler_faces = image_rescaler(image_normalizer_typ)

    # load dataset (shuffled, pre-processed, batched, and pre-fetched)
    # original shape = (218, 178, 3)
    resize_to = (32, 32)

    # Data cant be stored in repo, insert your own data dirr
    data_directory = "C:/Users/Jakob/OneDrive/Skrivbord/Skrivbord/DL_proj/celeba_gan/img_align_celeba"
    
    kwargs = {"data_directory": data_directory, 
            "resize_to": resize_to,
            "reduce_to": None,
            "normalizer": normalizer_faces}
    dataset_name = "faces"
    batch_size = 64
    dataset, dataset_size = data_pipeline_load(dataset_name, **kwargs)    
    dataset, dataset_size, dataset_fid, fid_size = data_pipeline_pre_train(dataset, dataset_size, batch_size, fid_split=0.1)

    # tensorboard logs are saved here
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_id = current_time + "-" + dataset_name
    log_dir = os.path.join('logs', experiment_id)
    os.makedirs(log_dir, exist_ok=True)

    train_log_dir = os.path.join(log_dir, "train")
    gen_log_dir = os.path.join(log_dir, "gen")
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    gen_summary_writer = tf.summary.create_file_writer(gen_log_dir)

    # training chekcpoints are saved here
    checkpoint_dir = os.path.join('training_checkpoints', experiment_id)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    # generated images are saved here
    viz_save_path = os.path.join("assets", experiment_id)
    os.makedirs(viz_save_path, exist_ok=True)

    # full saved models come here
    saved_generator_dir = os.path.join('saved_models', "generator", experiment_id)
    saved_generator_path = os.path.join(saved_generator_dir, "generator.h5")
    saved_discriminator_dir = os.path.join('saved_models', "discriminator", experiment_id)
    saved_discriminator_path = os.path.join(saved_discriminator_dir, "discriminator.h5")

    # make generator and discriminator
    latent_dim = 100
    input_shape = (latent_dim,)
    output_shape = (32, 32, 3)
    generator = make_generator_2_faces(input_shape, output_shape)
    discriminator = make_discriminator_2_faces(output_shape)

    # make optimizers
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

    # make training metrics
    gen_loss_metric = tf.keras.metrics.Mean('gen_loss_metric', dtype=tf.float32)
    disc_loss_metric = tf.keras.metrics.Mean('disc_loss_metric', dtype=tf.float32)

    # make checkpint saver callback
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)

    # cross-entropy helper
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # some config
    epochs = 500
    # as defined before for loading and batching the dataset
    assert batch_size == 64
    num_examples_to_generate = 16

    # You will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    seed = tf.random.normal([num_examples_to_generate, latent_dim])

    # FID
    use_fid = True

    if use_fid:
      fid_scorer = FID(rescaler=rescaler_faces)
      fid_score_metric = tf.keras.metrics.Mean('fid_score_metric', dtype=tf.float32)
      fid_dict = {
          "fid_scorer": fid_scorer, 
          "fid_score_metric": fid_score_metric,
          "dataset_fid": dataset_fid,
          "fid_size": fid_size
          }
      
    else:
      fid_dict = {}

    ckpt_save_epoch = 1

    # train
    train(dataset=dataset,
          gen_loss_metric=gen_loss_metric, 
          disc_loss_metric=disc_loss_metric, 
          train_summary_writer=train_summary_writer, 
          gen_summary_writer=gen_summary_writer, 
          epochs=epochs, 
          batch_size=batch_size, 
          latent_dim=latent_dim, 
          generator=generator, 
          discriminator=discriminator,
          cross_entropy=cross_entropy, 
          generator_optimizer=generator_optimizer, 
          discriminator_optimizer=discriminator_optimizer, 
          seed=seed, 
          viz_save_path=viz_save_path, 
          checkpoint_prefix=checkpoint_prefix, 
          dataset_name=dataset_name,
          dataset_size=dataset_size,
          rescaler=rescaler_faces, 
          ckpt_save_epoch=ckpt_save_epoch,
          **fid_dict)

    # save full models
    generator.save(saved_generator_path)
    discriminator.save(saved_discriminator_path)