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
from datapipeline import *
import time
import imageio
import glob
import tensorflow_docs.vis.embed as embed

# To install tensorflow_docs pip install -q git+https://github.com/tensorflow/docs


os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'  # minimize TF annoying messages

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Uncomment for cpu

# Make tf work on 3060 gpu

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

tf.config.run_functions_eagerly(True)

print(tf.__version__)   # Check your TF version (should be 2.4.0)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

DATA_DIRECTORY = '../data/celebA/images'

print(tf.__version__)
print(f"GPUs used: {tf.config.list_physical_devices('GPU')}")

def conv_out_size_same(size, stride):
    """ Faces loading pipeline.
    
    Parameters
    ----------
    size : int
        Size of dimension (height or width) of layer n.
    stride : int
        Stride size in that dimension of layer n.
    
    Returns
    -------
    int
        The size of the dimension (height or width) in layer n-1 to obtain same convoltuion
        from layer n-1 to n.
    
    Raises
    ------
    None
    
    Notes
    -----
    Source:
    https://github.com/carpedm20/DCGAN-tensorflow/blob/62c9a2a7f74505cad30858bf40307c93e5bd9293/model.py#L14
    """
    return int(math.ceil(float(size) / float(stride)))

class DCGAN(keras.Model):
    def __init__(self, input_shape, name, **cfg):
        super(DCGAN, self).__init__(name=name)
        #self.make_generator(input_shape)
        self.cfg = cfg
        self.gen_cfg = cfg["generator"]
        #self.disc_cfg = cfg["discriminator"]
    
    def complement_gen_cfg(self, gen_cfg, output_shape):
        """ Build config for generator. Config is yaml file. 
        
        Parameters
        ----------
        gen_cfg : dict
            Generator config dict from yaml file.
        output_shape : tuple
            Height, width, channels of output feature map.

        Returns
        -------
        dict
            gen_cfg but complemented by feature map shapes (heigh, width), most importantly
            with the project-reshape z shape.

        Raises
        ------
        None

        Notes
        -----
        E.g.: the architecture in the paper (gen_cfg):
        
        generator:
          pr:
            filters: 1024
          h1:
            filters: 512
            strides: [2,2]
            kernel_size: [5,5]
          h2:
            filters: 256
            strides: [2,2]
            kernel_size: [5,5]
          h3:
            filters: 128
            strides: [2,2]
            kernel_size: [5,5]
          h4:
            filters: 3
            strides: [2,2]
            kernel_size: [5,5]
        
        and output_shape is (64,64,3).
        Then, 
        
        s_h, s_w, s_c = output_shape = (64,64,3)
        
        s_c4 = 3 -> generator["h4"]["filters"]
        s_c3 = 512 -> generator["h3"]["filters"]
        s_c2 = 256 -> generator["h2"]["filters"]
        s_c1 = 512 -> generator["h1"]["filters"]
        s_c_pr = 1024 -> generator["pr"]["filters"]
        
        s_s4 = (2, 2) -> generator["h4"]["strides"]
        s_s3 = (2, 2) -> generator["h3"]["strides"]
        s_s2 = (2, 2) -> generator["h2"]["strides"]
        s_s1 = (2, 2) -> generator["h1"]["strides"]
        
        s_f4 = (5,5) -> generator["h4"]["kernel_size"]
        s_f3 = (5,5) -> generator["h3"]["kernel_size"]
        s_f2 = (5,5) -> generator["h2"]["kernel_size"]
        s_f1 = (5,5) -> generator["h1"]["kernel_size"]
        
        s_h3, s_w3 = conv_out_size_same(s_h, s_s4[0]), conv_out_size_same(s_w, s_s4[1]) = (32,32)
        s_h2, s_w2 = conv_out_size_same(s_h3, s_s3[0]), conv_out_size_same(s_w3, s_s3[1]) = (16,16)
        s_h1, s_w1 = conv_out_size_same(s_h2, s_s2[0]), conv_out_size_same(s_w2, s_s2[1]) = (8,8)
        s_h_pr, s_w_pr = conv_out_size_same(s_h1, s_s1[0]), conv_out_size_same(s_w1, s_s1[1]) = (4,4)
        
        where e.g.: s_h3, s_w3 is the feature map shape (h,w) in the 3rd upsampling conv2
        and s_h_pr, s_w_pr is the shape (h,w) of the project-reshape on z.
        Each hx layer is an upsampling conv2d layer.
        
        Note that upsampling conv2d is always same convolution.
        The whole point of this function is to obtain the feature map shapes and the project-reshape z
        shape from the output_shape, the number of filters (filters), the stride size (strides),
        and the kernel sizes.
        """
        # Get output shape.
        s_h, s_w, s_c = output_shape
        
        # Get the layer keys.
        keys = list(gen_cfg.keys())
        keys_reversed = list(reversed(keys))
        
        # Set temp variables.
        s_hx = s_h
        s_wx = s_w
        
        # Reverse iterate through layers.
        for idx, k in enumerate(keys_reversed):
            # if project-reshape z (layer no. 0), then set the last layer 
            # (layer before pr in reverse, last list element) feature map shape
            if k == "pr":
                gen_cfg[keys_reversed[0]]["s_h"] = s_h
                gen_cfg[keys_reversed[0]]["s_w"] = s_w
                # exit loop
                continue
            
            # get stride size
            s_sx = gen_cfg[k]["strides"]
            
            # get feature map shape
            s_hx, s_wx = \
                conv_out_size_same(s_hx, s_sx[0]), conv_out_size_same(s_wx, s_sx[1])
            
            # set the feature map shape in the layer before based on the current layer
            gen_cfg[keys_reversed[idx+1]]["s_h"] = s_hx
            gen_cfg[keys_reversed[idx+1]]["s_w"] = s_wx
            
        return deepcopy(gen_cfg) 
    
    def make_generator(self, input_shape, output_shape):
        """ Make the generator model. Note that depending on the dataset, the feature map shapes,
        kernel sizes, and channels will change. Encapsulated config in yaml file to automatically
        do this.
    
        Parameters
        ----------
        input_shape : tuple
            Input image shape, height, width, channels.
        output_shape : tuple
            Output image shape, height, width, channels.

        Returns
        -------
        keras.Model
            The generator model.

        Raises
        ------
        None

        Notes
        -----
        None
        """
        # complement the generator config with the feature map shapes, and
        # most importantly with the proejct-reshape z shape.
        gen_cfg = self.complement_gen_cfg(self.gen_cfg, output_shape)
        
        # take care of input z (!!)
        # Functional API for more flexibility
        inputs = keras.Input(shape=input_shape, name="generator_input")
        
        # NOTE: Number of filters should be adapted to each dataset <- FIND WAY TO DO IT AUTOMATICALLY
        # idea: define build_mnist_generator() and use that in if else
        # project `z` and reshape
        s_h_pr = gen_cfg["pr"]["s_h"]
        s_w_pr = gen_cfg["pr"]["s_w"]
        s_c_pr = gen_cfg["pr"]["filters"]
        z_proj = layers.Dense(s_h_pr*s_w_pr*s_c_pr, use_bias=False, name="g_h0_lin")(inputs)
        h = layers.Reshape((s_h_pr, s_w_pr, s_c_pr), name="g_h0")(z_proj)
        h = layers.BatchNormalization(name="g_h0_bn")(h)
        h = layers.LeakyReLU(name="g_h0_a")(h)
        
        # Note: None is the batch size
        assert tuple(h.shape) == (None, s_h_pr, s_w_pr, s_c_pr)
        
        # get layer keys
        keys = list(gen_cfg.keys())
        
        # iterate through the layers from the beginning
        # skip project-reshape z, and deal with last upsample conv2d later
        for k in keys[1:-1]:
            # upsample conv2d layer
            s_c = gen_cfg[k]["filters"]
            s_s = gen_cfg[k]["strides"]
            s_f = gen_cfg[k]["kernel_size"]
            
            h = layers.Conv2DTranspose(filters=s_c, 
                                       kernel_size=s_f, 
                                       strides=s_s, 
                                       padding='same', 
                                       use_bias=False, 
                                       name=f"g_{k}")(h)

            h = layers.BatchNormalization(name=f"g_{k}_bn")(h)
            h = layers.LeakyReLU(name=f"g_{k}_a")(h)
            
            s_h = gen_cfg[k]["s_h"]
            s_w = gen_cfg[k]["s_w"]
            
            assert tuple(h.shape) == (None, s_h, s_w, s_c)

        # upsample conv2d last layer
        k = keys[-1]
        s_c = gen_cfg[k]["filters"]
        s_s = gen_cfg[k]["strides"]
        s_f = gen_cfg[k]["kernel_size"]
        
        h = layers.Conv2DTranspose(filters=s_c, 
                                   kernel_size=s_f, 
                                   strides=s_s, 
                                   padding='same', 
                                   use_bias=False, 
                                   activation='tanh',
                                   name=f"g_{k}")(h)
        
        s_h = gen_cfg[k]["s_h"]
        s_w = gen_cfg[k]["s_w"]
        
        assert tuple(h.shape) == (None, s_h, s_w, s_c)
        
        self.generator = tf.keras.Model(inputs=inputs, outputs=h, name="generator")
        
        return self.generator

    def make_discriminator(self, output_shape):
        """ Make the discriminator model. Model is fixed (at least for now).
        Can go down from arbitrary output_shape to sigmoid prediciton.
    
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
        None
        """
        outputs = keras.Input(shape=output_shape, name="discriminator_input")
        
        h = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',name="d_h0")(outputs)
        h = layers.LeakyReLU()(h)
        h = layers.Dropout(0.3)(h)

        h = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', name="d_h1")(h)
        h = layers.LeakyReLU()(h)
        h = layers.Dropout(0.3)(h)

        h = layers.Flatten(name="d_h2_lin")(h)
        h = layers.Dense(1)(h)
        
        self.discriminator = tf.keras.Model(inputs=outputs, outputs=h, name="discriminator")

        return self.discriminator
    
    def evaluate_loss(self,real_images,fake_images,_type):
        '''
                Generator -> Goal is to generate the "realest" fake images, so the loss is only measured taking those in consideration.

                Discriminator -> We want the output to be 1 for real images and 0 for fake images. 
                Here, the total loss will be aggregated loss for real and fake images (which is obvious because we're trying 
                to see how these compare)

        '''
        f_loss = keras.losses.BinaryCrossentropy(from_logits=True)
        if _type == 'generator':
            return f_loss(tf.ones_like(fake_images),fake_images)   # transform fake images array into an array of ones
        elif _type == 'discriminator':
            return f_loss(tf.ones_like(real_images),real_images) + f_loss(tf.zeros_like(fake_images),fake_images)

    def optimize(self, learning_rate=0.0002,beta1=0.5):
        '''
            According to the original DCGAN paper(*): 
                1) Learning rate - " While previous GAN work has used momentum to accelerate training, 
            we used the Adam optimizer (Kingma & Ba, 2014) with tuned hyperparameters. We found the suggested learning 
            rate of 0.001, to be too high, using 0.0002 instead."

                2) Momentum (beta1) - "Additionally, we found leaving the momentum term β1 at the suggested value of 0.9 resulted
                 in training oscillation and instability while reducing it to 0.5 helped stabilize training.


            (*) https://arxiv.org/pdf/1511.06434.pdf


            Output:
                Optimizer for both generator and discriminator model
        '''
        return keras.optimizers.Adam(learning_rate,beta_1=beta1), keras.optimizers.Adam(learning_rate,beta_1=beta1)

    def step(self, images, batch_size, noise_dim, old_generator, old_discriminator, learning_rate, beta_1):
        '''
        Method for running one single training step
        
        Parameters
        ----------
        images : tensor
            One batch of data

        batch_size : int
            batchsize
        
        noise_dim : int
            size of noise dimension

        old_generator : tf object
            generator from previous step

        old_discriminator : tf object
            discriminator from previous step

        learning_rate : float
            learning rate in the optimizer

        beta_1 : float
            beta parameter in the optimizer

        Returns
        -------
        generator
            the generator of the model at the current iteration 

        discriminator
            the discriminator of the model at the current iteration 

        '''
        noise = tf.random.normal([batch_size, noise_dim])
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = self.evaluate_loss(real_output, fake_output, 'generator')
            disc_loss = self.evaluate_loss(real_output, fake_output, 'discriminator')
            
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizier, discriminator_optimizer = self.optimize(learning_rate=learning_rate,beta1=beta_1)

        generator_optimizier.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        return generator, discriminator

    def fit(self, data, epochs, batch_size, noise_dim, old_generator, old_discriminator, seed, learning_rate, beta_1):
        '''
        Method for fitting the model
        
        Parameters
        ----------
        data : PrefectDataset
            The whole dataset

        epochs : int
            number of epochs to be ran
        
        noise_dim : int
            size of noise dimension

        old_generator : tf object
            generator from previous step

        old_discriminator : tf object
            discriminator from previous step

        seed : tf object
            seed to produce the first random image

        learning_rate : float
            learning rate in the optimizer

        beta_1 : float
            beta parameter in the optimizer

        Returns
        -------
        None
        '''
        for epoch in range(epochs):
            start = time.time()
            for image_batch in data:
                new_generator, new_discriminator = self.step(image_batch, batch_size, noise_dim, old_generator, old_discriminator, learning_rate, beta_1)
                old_generator = new_generator
                old_discriminator = new_generator

            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

            self.gen_imgs(old_generator, epoch + 1, seed)


    def gen_imgs(self, model, epoch, test_input):
        '''
        Method for generating images from the model at every epoch
        
        Parameters
        ----------
        model : tf object
            the generator of the model at the current epoch

        epochs : int
            number of epochs to be ran
        
        test_input : tf object
            seed to produce the first random image

        Returns
        -------
        None
        '''
        preds = model(test_input, training=False)

        fig = plt.figure(figsize=(4,4))

        for i in range(preds.shape[0]):
            plt.subplot(4,4,i+1)
            plt.imshow(preds[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')
        
        plt.savefig('imgs/images_at_epochs{:04d}.png'.format(epoch))
        #plt.show()
    
    def make_gif(self, gif_name):
        anim_file = gif_name + '.gif'

        with imageio.get_writer(anim_file, mode='I') as writer:
            filenames = glob.glob('image*.png')
            filenames = sorted(filenames)
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
            image = imageio.imread(filename)
            writer.append_data(image)

        embed.embed_file(anim_file)


if __name__ == "__main__":
    # Contributed: Diogo Pinheiro, Jakob Lindén, Márk Csizmadia, Patrick Jonsson
    
    # original shape = (28, 28, 1)
    resize_to = (32, 32)
    resize_to = None
    kwargs = {"data_directory": None, "resize_to": resize_to}
    dataset_name = "mnist"
    batch_size = 256

    # get data from pipeline
    dataset, dataset_size = data_pipeline_load(dataset_name, **kwargs)
    dataset = data_pipeline_pre_train(dataset, dataset_size, batch_size)

    # Tf example on mnist
    with open("config_tf_example.yml", 'r') as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # dims
    input_shape = (100,)
    output_shape = (28, 28, 1)

    # parameter values
    noise_dim = 100
    num_examples_to_generate = 16
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    epochs = 50
    learning_rate = 0.0002
    beta_1 = 0.5

    # create model
    dcgan = DCGAN(input_shape=input_shape, name="my_dcgan", **cfg)

    generator = dcgan.make_generator(input_shape, output_shape)
    discriminator = dcgan.make_discriminator(output_shape)

    # fit model
    dcgan.fit(dataset, epochs, batch_size, noise_dim, generator, discriminator, seed, learning_rate, beta_1)

    # make gif
    dcgan.make_gif('dcgan')