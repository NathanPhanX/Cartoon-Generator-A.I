import numpy as np
import glob
import tensorflow.compat.v1 as tf_version
import cv2
import time
import keras.backend as K
import random
import multiprocessing as mp
import matplotlib.pyplot as plt

from PIL import Image
from keras.initializers import RandomNormal
from keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D, Concatenate, Add, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.merge import _Merge
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import img_to_array, load_img
from functools import partial
from numba import cuda
from skimage.transform import resize

tf_version.disable_v2_behavior()  # Tensorflow 2.0 cannot be used to run this code


def frame_to_video():
    frames = []
    path = glob.glob("N:\\CODE\\DeepLearning\\Generated_Images_5\\*.png")

    for image_path in path:
        img = cv2.imread(image_path)
        frames.append(img)

    for i in range(len(frames)):
        cv2.imshow('video', frames[i])
        time.sleep(0.5)

        if cv2.waitKey(1) and 0XFF == ord('q'):
            break

    cv2.waitKey()
    cv2.destroyAllWindows()


class DC_GAN(object):
    def __init__(self):
        self.GENERATE_RES = 9  # Resolution of image is the multiple of 16
        self.GENERATE_SQUARE = 16 * self.GENERATE_RES  # Row and column of images
        self.IMAGE_CHANNEL = 3  # 1 for grayscale and 3 for RGB
        self.IMAGE_SHAPE = (self.GENERATE_SQUARE, self.GENERATE_SQUARE, self.IMAGE_CHANNEL)

        self.PREVIEW_ROW = 1  # Number of images per row when preview images
        self.PREVIEW_COLUMN = 1  # Number of images per column when preview images
        self.PREVIEW_MARGIN = 16  # Space between images
        self.SAVE_FREQ = 100  # frequency of saving images during training

        self.SEED_SIZE = self.GENERATE_SQUARE * self.GENERATE_SQUARE  # Size vector to generate images from

        self.EPOCH = 25000  # Number of epoch for training
        self.BATCH_SIZE = 24  # Size of image batches for training

    def build_generator(self):
        model = Sequential()

        model.add(Dense(2 * 2 * 1024, activation="relu", input_dim=self.SEED_SIZE))
        model.add(Reshape((2, 2, 1024)))

        model.add(UpSampling2D(size=2))
        model.add(Conv2D(1024, kernel_size=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D(size=2))
        model.add(Conv2D(512, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D(size=2))
        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D(size=3))
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D(size=3))
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Conv2D(self.IMAGE_CHANNEL, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))

        print(model.summary())

        input_model = Input(shape=(self.SEED_SIZE,))
        generate_image = model(input_model)

        return Model(input_model, generate_image)

    def build_discriminator(self):
        model = Sequential()

        model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=self.IMAGE_SHAPE, padding='same'))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=1, padding='same'))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=5, strides=3, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Conv2D(512, kernel_size=5, strides=3, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Conv2D(1024, kernel_size=7, strides=4, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        print(model.summary())

        input_image = Input(shape=self.IMAGE_SHAPE)

        validates = model(input_image)

        return Model(input_image, validates)

    def save_images(self, noise, generator):
        save_path = "N:\\CODE\\DeepLearning\\Generated_Images_4\\"
        path = glob.glob(save_path + "*.png")

        image_array = np.full((
            self.PREVIEW_MARGIN + (self.PREVIEW_ROW * (self.GENERATE_SQUARE + self.PREVIEW_MARGIN)),
            self.PREVIEW_MARGIN + (self.PREVIEW_COLUMN * (self.GENERATE_SQUARE + self.PREVIEW_MARGIN)), 3),
            255, dtype=np.uint8)

        generated_images = generator.predict(noise)
        generated_images = 0.5 * generated_images + 0.5

        image_count = 0
        for row in range(self.PREVIEW_ROW):
            for col in range(self.PREVIEW_COLUMN):
                r = row * (self.GENERATE_SQUARE + 16) + self.PREVIEW_MARGIN
                c = col * (self.GENERATE_SQUARE + 16) + self.PREVIEW_MARGIN
                image_array[r:r+self.GENERATE_SQUARE, c:c+self.GENERATE_SQUARE] = generated_images[image_count] * 255
                image_count += 1

        result = Image.fromarray(image_array)
        result.save(save_path + str(len(path)) + ".png")

    def load_data(self):
        data_path = glob.glob("N:\\CODE\\DeepLearning\\Anime_Images\\*.png")
        training_data = []

        for index in range(len(data_path)):
            image = Image.open(data_path[index]).resize((self.GENERATE_SQUARE, self.GENERATE_SQUARE), Image.ANTIALIAS)
            training_data.append(np.asanyarray(image))

        training_data = np.reshape(training_data, (-1, self.GENERATE_SQUARE, self.GENERATE_SQUARE, self.IMAGE_CHANNEL))
        training_data = training_data / 127.5 - 1.0  # Make the image which is originally between 0 and 255 into between -1 and 1

        y_real = np.ones((self.BATCH_SIZE, 1))
        y_fake = np.zeros((self.BATCH_SIZE, 1))

        return training_data, y_real, y_fake

    def train(self):
        optimizer = Adam(1.5e-4, 0.5)  # The value is taken from research paper

        discriminator = self.build_discriminator()  # Build the discriminator model
        discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])  # Compile the discriminator model
        generator = self.build_generator()  # Build the generator model

        random_input = Input(shape=(self.SEED_SIZE,))  # Create a random input
        generated_image = generator(random_input)  # Create generated image

        discriminator.trainable = False  # Prevent the weights to be adjusted

        validity = discriminator(generated_image)  # Create validity

        combined = Model(random_input, validity)  # Create a combine model
        combined.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])  # Compile a combine model

        fixed_seed = np.random.normal(0, 1, (self.PREVIEW_ROW * self.PREVIEW_COLUMN, self.SEED_SIZE))  # Seed for the saved images

        training_data, y_real, y_fake = self.load_data()

        for epoch in range(self.EPOCH):
            idx = np.random.randint(0, training_data.shape[0], self.BATCH_SIZE)
            x_real = training_data[idx]

            # Generate some images
            seed = np.random.normal(0, 1, (self.BATCH_SIZE, self.SEED_SIZE))
            x_fake = generator.predict(seed)

            # Train discriminator on real and fake
            discriminator_metric_real = discriminator.train_on_batch(x=x_real, y=y_real)
            discriminator_metric_generated = discriminator.train_on_batch(x=x_fake, y=y_fake)
            discriminator_metric = 0.5 * np.add(discriminator_metric_real, discriminator_metric_generated)

            # Train the generator on calculate loss
            generator_metric = combined.train_on_batch(seed, y_real)

            # Save the image
            if epoch % self.SAVE_FREQ == 0:
                self.save_images(fixed_seed, generator)
                print(f"Epoch {epoch}, Discriminator accuracy: {discriminator_metric[1]}, Generator accuracy: {generator_metric[1]}")

        generator_path = "C:\\Users\\phuoc\\OneDrive\\Desktop\\CODE\\DeepLearning\\Model\\Anime_Generator.h5"
        generator.save(generator_path)


'''
class W_GAN(object):
    class RandomWeightedAverage(_Merge):
        """Provides a (random) weighted average between real and generated image samples"""

        def _merge_function(self, inputs):
            alpha = K.random_uniform((32, 1, 1, 1))
            return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

    def __init__(self):
        self.PREVIEW_ROW = 10  # Number of images per row when preview images
        self.PREVIEW_COLUMN = 10  # Number of images per column when preview images
        self.PREVIEW_MARGIN = 16  # Space between images

        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = self.img_rows * self.img_cols

        self.batch_size = 32
        self.epochs = 10000
        self.sample_interval = 100

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = RMSprop(lr=0.00005)

        # Build the generator and critic
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)

        # Noise input
        z_disc = Input(shape=(self.latent_dim,))

        # Generate image based of noise (fake sample)
        fake_img = self.generator(z_disc)

        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = self.RandomWeightedAverage()([real_img, fake_img])

        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss, averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty'  # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc], outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.Wasserstein_loss, self.Wasserstein_loss, partial_gp_loss], optimizer=optimizer, loss_weights=[1, 1, 10])

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.latent_dim,))

        # Generate images based of noise
        img = self.generator(z_gen)

        # Discriminator determines validity
        valid = self.critic(img)

        # Defines generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.Wasserstein_loss, optimizer=optimizer)

    def load_data(self):
        data_path = glob.glob("N:\\CODE\\DeepLearning\\Anime_Images\\*.png")
        training_data = []

        for index in range(len(data_path)):
            image = Image.open(data_path[index]).resize((self.img_rows, self.img_cols), Image.ANTIALIAS)
            training_data.append(np.asanyarray(image))

        training_data = np.reshape(training_data, (-1, self.img_rows, self.img_cols, self.channels))
        training_data = training_data.astype('float32')
        training_data = training_data / 127.5 - 1.0  # Make the image which is originally between 0 and 255 into between -1 and 1

        return training_data

    @staticmethod
    def gradient_penalty_loss(y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]

        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)

        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))

        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)

        # compute lambda * (1 - ||grad||)^2 still for each single sample

        gradient_penalty = K.square(1 - gradient_l2_norm)

        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)

    @staticmethod
    def Wasserstein_loss(y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):
        model = Sequential()

        model.add(Dense(1 * 1 * 1024, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((1, 1, 1024)))

        model.add(UpSampling2D(size=2))
        model.add(Conv2D(1024, kernel_size=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D(size=2))
        model.add(Conv2D(1024, kernel_size=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D(size=2))
        model.add(Conv2D(512, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D(size=2))
        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D(size=2))
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D(size=2))
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Conv2D(self.channels, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):
        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=1, padding='same'))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=1, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=5, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=5, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Conv2D(512, kernel_size=7, strides=3, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Conv2D(1024, kernel_size=7, strides=3, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='linear'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self):
        # Load the data
        x_train = self.load_data()

        # Seed for the saved images
        fixed_seed = np.random.normal(0, 1, (self.PREVIEW_ROW * self.PREVIEW_COLUMN, self.latent_dim))

        # Adversarial ground truths
        valid = -np.ones((self.batch_size, 1))
        fake = np.ones((self.batch_size, 1))
        dummy = np.zeros((self.batch_size, 1))  # Dummy gt for gradient penalty

        for epoch in range(self.epochs):
            for _ in range(self.n_critic):
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, x_train.shape[0], self.batch_size)
                imgs = x_train[idx]

                # Sample generator input
                noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))

                # Train the critic
                d_loss = self.critic_model.train_on_batch([imgs, noise], [valid, fake, dummy])

            # ---------------------
            #  Train Generator
            # ---------------------
            g_loss = self.generator_model.train_on_batch(noise, valid)

            # If at save interval => save generated image samples
            if epoch % self.sample_interval == 0:
                print("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))
                self.save_images(fixed_seed, self.generator)

    def save_images(self, noise, generator):
        save_path = "N:\\CODE\\DeepLearning\\Generated_Images_4\\"
        path = glob.glob(save_path + "*.jpg")

        image_array = np.full((
            self.PREVIEW_MARGIN + (self.PREVIEW_ROW * (self.img_rows + self.PREVIEW_MARGIN)),
            self.PREVIEW_MARGIN + (self.PREVIEW_COLUMN * (self.img_rows + self.PREVIEW_MARGIN)), 3),
            255, dtype=np.uint8)

        generated_images = generator.predict(noise)
        generated_images = 0.5 * generated_images + 0.5

        image_count = 0
        for row in range(self.PREVIEW_ROW):
            for col in range(self.PREVIEW_COLUMN):
                r = row * (self.img_rows + 16) + self.PREVIEW_MARGIN
                c = col * (self.img_rows + 16) + self.PREVIEW_MARGIN
                image_array[r:r + self.img_rows, c:c + self.img_rows] = generated_images[image_count] * 255
                image_count += 1

        result = Image.fromarray(image_array)
        result.save(save_path + str(len(path)) + ".jpg")
'''
'''
# weighted sum output for ProgressiveGrowingGan
class WeightedSum(Add):
    def __init__(self, alpha=0.0, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)
        self.alpha = K.variable(alpha, name='ws_alpha')

    # output a weighted sum of inputs
    def _merge_function(self, inputs):
        # only supports a weighted sum of two inputs
        assert (len(inputs) == 2)

        # output = ((1-a) * input1) + (a * input2)
        output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
        return output


class ProgressiveGrowingGan(object):
    def __init__(self):
        self.PREVIEW_ROW = 5  # Number of images per row when preview images
        self.PREVIEW_COLUMN = 5  # Number of images per column when preview images
        self.PREVIEW_MARGIN = 16  # Space between images
        self.save_freq = 100

    def add_disciminator_block(self, old_model, num_input_layers=3):
        # Get the shape of the existing model
        model_shape = list(old_model.input.shape)

        # Define new shape as double the size
        input_shape = (model_shape[-2].value * 2, model_shape[-2].value * 2, model_shape[-1].value)
        input_img = Input(shape=input_shape)

        # Define new input processing layer
        d = Conv2D(64, (1, 1), padding='same', kernel_initializer='he_normal')(input_img)
        d = LeakyReLU(alpha=0.2)(d)

        # Define new block
        d = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = AveragePooling2D()(d)
        new_block = d

        # Skip all above for the old model
        for i in range(num_input_layers, len(old_model.layers)):
            d = old_model.layers[i](d)

        # Define straight-through model
        model1 = Model(input_img, d)

        # Compile the model1
        model1.compile(loss='mse', optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

        # Downsample larger images
        downsample = AveragePooling2D()(input_img)

        # Connect old input processing to the downsample new input
        old_block = old_model.layers[1](downsample)
        old_block = old_model.layers[2](old_block)

        # Fade in the output of the old model input layer with new input
        d = WeightedSum()([old_block, new_block])

        # SKip the input, 1x1 and activation for the old model
        for i in range(num_input_layers, len(old_model.layers)):
            d = old_model.layers[i](d)

        # Define straight-through model
        model2 = Model(input_img, d)

        # Compile model
        model2.compile(loss='mse', optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

        return [model1, model2]

    def define_discriminator(self, num_block, input_shape=(4, 4, 3)):
        model_list = list()

        # Base model input
        input_image = Input(shape=input_shape)

        d = Conv2D(64, (1, 1), padding='same', kernel_initializer='he_normal')(input_image)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(128, (4, 4), padding='same', kernel_initializer='he_normal')(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Flatten()(d)
        output_class = Dense(1)(d)

        # Define model
        model = Model(input_image, output_class)

        # Compile model
        model.compile(loss='mse', optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

        # Store model
        model_list.append(([model, model]))

        # Create sub_model
        for i in range(1, num_block):
            old_model = model_list[i-1][0]
            models = self.add_disciminator_block(old_model)
            model_list.append(models)

        return model_list

    def add_generator_block(self, old_model):
        # Get the end of last block
        block_end = old_model.layers[-2].output

        # Upsample and define a new block
        upsample = UpSampling2D()(block_end)

        g = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(upsample)
        g = BatchNormalization()(g)
        g = LeakyReLU(alpha=0.2)(g)

        g = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(g)
        g = BatchNormalization()(g)
        g = LeakyReLU(alpha=0.2)(g)

        # Add new output layer
        output_image = Conv2D(3, (1, 1), padding='same', kernel_initializer='he_normal')(g)

        # Define model
        model1 = Model(old_model.input, output_image)

        # Get the output layer from the old model
        output_old_model = old_model.layers[-1]

        # Connect the upsampling to the old output layer
        output_image2 = output_old_model(upsample)

        # Define new output image as the weighted sum of the old and new models
        merged = WeightedSum()([output_image2, output_image])

        # Define model
        model2 = Model(old_model.input, merged)

        return [model1, model2]

    def define_generator(self, latent_dim, num_block, input_dim=4):
        model_list = list()

        # Define input
        input_latent = Input(shape=(latent_dim,))

        # Define model
        g = Dense(input_dim * input_dim * 128, kernel_initializer='he_normal')(input_latent)
        g = Reshape((input_dim, input_dim, 128))(g)

        g = Conv2D(128, (4, 4), padding='same', kernel_initializer='he_normal')(g)
        g = BatchNormalization()(g)
        g = LeakyReLU(alpha=0.2)(g)

        g = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(g)
        g = BatchNormalization()(g)
        g = LeakyReLU(alpha=0.2)(g)

        output_img = Conv2D(3, (1, 1), padding='same', kernel_initializer='he_normal')(g)
        model = Model(input_latent, output_img)

        # Store the model
        model_list.append([model, model])

        # Create sub_models
        for i in range(1, num_block):
            old_model = model_list[i-1][0]
            models = self.add_generator_block(old_model)
            model_list.append(models)

        return model_list

    def define_composite_model(self, discriminators, generators):
        model_list = list()

        # Create composite models
        for i in range(len(discriminators)):
            g_models, d_models = generators[i], discriminators[i]

            # Straight-through model
            d_models[0].trainable = False
            model1 = Sequential()
            model1.add(g_models[0])
            model1.add(d_models[0])
            model1.compile(loss='mse', optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

            # Fade-in model
            d_models[1].trainable = False
            model2 = Sequential()
            model2.add(g_models[1])
            model2.add(d_models[1])
            model2.compile(loss='mse', optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

            # Store both models
            model_list.append([model1, model2])

        return model_list

    def update_fade_in(self, models, step, num_steps):
        # Calculate current alpha (linear from 0 to 1)
        alpha = step / float(num_steps - 1)

        # Update the alpha value for each model
        for model in models:
            for layer in model.layers:
                if isinstance(layer, WeightedSum):
                    K.set_value(layer.alpha, alpha)

    def scale_image(self, images, new_shape):
        image_list = list()

        for image in images:
            new_img = resize(image, new_shape, 0)
            image_list.append(new_img)

        return np.asarray(image_list)

    def generate_real_sample(self, data_set, batch_size):
        index = np.random.randint(0, data_set.shape[0], batch_size)
        x_real = data_set[index]
        y_real = np.ones((batch_size, 1))
        return x_real, y_real

    def generate_fake_sample(self, g_model, latent_dim, batch_size):
        x_fake = g_model.predict(latent_dim)
        y_fake = -np.ones((batch_size, 1))
        return x_fake, y_fake

    def train_epoch(self, g_model, d_model, gan_model, data_set, latent_dim, num_epoch, num_batch, fixed_seed, fadein=False):
        batch_per_epoch = int(data_set.shape[0] / num_batch)
        num_step = batch_per_epoch * num_epoch
        half_batch = int(num_batch / 2)
        print(num_step)

        for i in range(num_step):
            if fadein:
                self.update_fade_in([g_model, d_model, gan_model], i, num_step)

            seed = np.random.normal(0, 1, (half_batch, latent_dim))

            x_real, y_real = self.generate_real_sample(data_set, half_batch)
            x_fake, y_fake = self.generate_fake_sample(g_model, seed, half_batch)

            d_loss1 = d_model.train_on_batch(x_real, y_real)
            d_loss2 = d_model.train_on_batch(x_fake, y_fake)
            g_loss = gan_model.train_on_batch(seed, y_real)

            # Save the images
            if i % self.save_freq == 0:
                self.save_images(fixed_seed, g_model)
                print('>%d, d1=%.3f, d2=%.3f g=%.3f' % (i, d_loss1, d_loss2, g_loss))

    def load_data(self):
        data_path = glob.glob("N:\\CODE\\DeepLearning\\Anime_Images\\*.png")
        training_data = []

        for index in range(len(data_path)):
            image = Image.open(data_path[index]).resize((64, 64), Image.ANTIALIAS)
            training_data.append(np.asanyarray(image))

        training_data = np.reshape(training_data, (-1, 64, 64, 3))
        training_data = training_data / 127.5 - 1.0  # Make the image which is originally between 0 and 255 into between -1 and 1

        return training_data

    def save_images(self, noise, generator):
        save_path = "N:\\CODE\\DeepLearning\\Generated_Images_5\\"
        path = glob.glob(save_path + "*.png")

        generated_images = generator.predict(noise)
        generated_images = 0.5 * generated_images + 0.5

        shape = generated_images.shape[1]

        image_array = np.full((
            self.PREVIEW_MARGIN + (self.PREVIEW_ROW * (shape + self.PREVIEW_MARGIN)),
            self.PREVIEW_MARGIN + (self.PREVIEW_COLUMN * (shape + self.PREVIEW_MARGIN)), 3),
            255, dtype=np.uint8)

        image_count = 0
        for row in range(self.PREVIEW_ROW):
            for col in range(self.PREVIEW_COLUMN):
                r = row * (shape + 16) + self.PREVIEW_MARGIN
                c = col * (shape + 16) + self.PREVIEW_MARGIN
                image_array[r:r + shape, c:c + shape] = generated_images[image_count] * 255
                image_count += 1

        result = Image.fromarray(image_array)
        result.save(save_path + str(len(path)) + ".png")

    def train(self, g_models, d_models, gan_models, data_set, latent_dim, e_norm, e_fade_in, num_batch):
        # Fit the base line model
        g_normal, d_normal, gan_normal = g_models[0][0], d_models[0][0], gan_models[0][0]

        # Scale the data set to the appropriate size
        gen_shape = g_normal.output_shape
        scaled_data = self.scale_image(data_set, gen_shape[1:])

        fixed_seed = np.random.normal(0, 1, (self.PREVIEW_ROW * self.PREVIEW_COLUMN, latent_dim))

        # Train the normal models
        self.train_epoch(g_normal, d_normal, gan_normal, scaled_data, latent_dim, e_norm, num_batch, fixed_seed)
        print('checkpoint1')

        # Process each level of growth
        for i in range(1, len(g_models)):
            [g_normal, g_fade_in] = g_models[i]
            [d_normal, d_fade_in] = d_models[i]
            [gan_normal, gan_fade_in] = gan_models[i]

            # Scale the data set to the appropriate size
            gen_shape = g_normal.output_shape
            scaled_data = self.scale_image(data_set, gen_shape[1:])

            # Train fade-in models for the next level of growth
            self.train_epoch(g_fade_in, d_fade_in, gan_fade_in, scaled_data, latent_dim, e_fade_in, num_batch, fixed_seed, True)

            # Train normal models
            self.train_epoch(g_normal, d_normal, gan_normal, scaled_data, latent_dim, e_norm, num_batch, fixed_seed)

        return g_normal

    def run(self):
        num_block = 5
        latent_dim = 500

        d_models = self.define_discriminator(num_block)
        g_models = self.define_generator(latent_dim, num_block)
        gan_model = self.define_composite_model(d_models, g_models)

        data_set = self.load_data()
        model = self.train(g_models, d_models, gan_model, data_set, latent_dim, 20, 20, 64)
        model.save("C:\\Users\\phuoc\\OneDrive\\Desktop\\CODE\\DeepLearning\\Model\\Advanced_Anime_Generator.h5")
'''


class ProgressiveGrowingGan1(object):
    def __init__(self):
        self.IMAGE_CHANNEL = 3  # 1 for grayscale and 3 for RGB
        self.IMAGE_SHAPE = (4, 4, 3)  # Image shape
        self.NUM = 6  # Resolution of final images will be 2^(NUM+1)
        self.LAYERS = 128  # Layers for convolutional neural network

        self.PREVIEW_ROW = 1  # Number of images per row when preview images
        self.PREVIEW_COLUMN = 1  # Number of images per column when preview images
        self.PREVIEW_MARGIN = 16  # Space between images
        self.SAVE_FREQ = 100  # frequency of saving images during training

        self.SEED_SIZE = 100  # Size vector to generate images from

        self.EPOCH_PER_NUM = 20000  # Number of epoch per resolution for training
        self.BATCH_SIZE = 32  # Size of image batches for training

    def build_generator(self):
        input_latent = Input(shape=(self.SEED_SIZE,))

        x = Dense(2 * 2 * 128, activation='relu', input_dim=self.SEED_SIZE)(input_latent)
        x = Reshape((2, 2, 128))(x)

        x = UpSampling2D()(x)
        x = Conv2D(self.LAYERS, kernel_size=2, padding='same')(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation('relu')(x)

        x = Conv2D(self.IMAGE_CHANNEL, kernel_size=2, padding='same')(x)
        output_img = Activation('tanh')(x)

        generator = Model(input_latent, output_img)
        return generator

    def build_new_generator(self, generator):
        old_model = generator.layers[-3].output

        x = UpSampling2D()(old_model)
        x = Conv2D(self.LAYERS, kernel_size=2, padding='same')(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation('relu')(x)

        x = Conv2D(self.IMAGE_CHANNEL, kernel_size=2, padding='same')(x)
        x = Activation('tanh')(x)

        new_generator = Model(generator.get_input_at(0), x)
        return new_generator

    def build_discriminator(self):
        input_image = Input(shape=self.IMAGE_SHAPE)

        x = Conv2D(self.LAYERS, kernel_size=2, strides=2, padding='same', input_shape=self.IMAGE_SHAPE)(input_image)
        x = LeakyReLU(alpha=0.2)(x)

        x = Dropout(0.25)(x)
        x = Conv2D(self.LAYERS, kernel_size=2, strides=2, padding='same')(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Flatten()(x)
        x = Dense(1, activation='sigmoid')(x)

        discriminator = Model(input_image, x)
        return discriminator

    def build_new_discriminator(self, discriminator, old_shape):
        input_shape = (2**old_shape, 2**old_shape, 3)
        input_img = Input(shape=input_shape)

        x = Conv2D(self.LAYERS, kernel_size=2, strides=2, padding='same', input_shape=input_shape)(input_img)
        x = LeakyReLU(alpha=0.2)(x)

        for i in range(3, len(discriminator.layers) - 2):
            x = discriminator.layers[i](x)

        x = Dropout(0.25)(x)
        x = Conv2D(self.LAYERS, kernel_size=2, strides=2, padding='same')(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Flatten()(x)
        x = Dense(1, activation='sigmoid')(x)

        new_discriminator = Model(input_img, x)
        return new_discriminator

    def save_images(self, noise, generator):
        save_path = "N:\\CODE\\DeepLearning\\Generated_Images_5\\"
        path = glob.glob(save_path + "*.png")

        generated_images = generator.predict(noise)
        generated_images = 0.5 * generated_images + 0.5

        shape = generated_images.shape[1]

        image_array = np.full((
            self.PREVIEW_MARGIN + (self.PREVIEW_ROW * (shape + self.PREVIEW_MARGIN)),
            self.PREVIEW_MARGIN + (self.PREVIEW_COLUMN * (shape + self.PREVIEW_MARGIN)), 3),
            255, dtype=np.uint8)

        image_count = 0
        for row in range(self.PREVIEW_ROW):
            for col in range(self.PREVIEW_COLUMN):
                r = row * (shape + 16) + self.PREVIEW_MARGIN
                c = col * (shape + 16) + self.PREVIEW_MARGIN
                image_array[r:r+shape, c:c+shape] = generated_images[image_count] * 255
                image_count += 1

        result = Image.fromarray(image_array)
        result.save(save_path + str(len(path)) + ".png")

    def load_data(self, num):
        data_path = glob.glob("N:\\CODE\\DeepLearning\\Anime_Images\\*.png")
        training_data = np.zeros((len(data_path), 2**num, 2**num, self.IMAGE_CHANNEL))

        for i in range(len(data_path)):
            training_data[i] = img_to_array(load_img(data_path[i], color_mode="rgb", target_size=(2**num, 2**num)))

        training_data = training_data.astype('float16')
        training_data = training_data / 127.5 - 1.0

        y_real = np.ones((self.BATCH_SIZE, 1))
        y_fake = np.zeros((self.BATCH_SIZE, 1))

        return training_data, y_real, y_fake

    def train(self):
        optimizer = Adam(1.5e-4, 0.5)  # The value is taken from research paper

        discriminator = self.build_discriminator()  # Build the discriminator model
        discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])  # Compile the discriminator model
        generator = self.build_generator()  # Build the generator model

        random_input = Input(shape=(self.SEED_SIZE,))  # Create a random input
        generated_image = generator(random_input)  # Create generated image

        discriminator.trainable = False  # Prevent the weights to be adjusted
        validity = discriminator(generated_image)  # Create validity

        combined = Model(random_input, validity)  # Create a combine model
        combined.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])  # Compile a combine model

        fixed_seed = np.random.normal(0, 1, (self.PREVIEW_ROW * self.PREVIEW_COLUMN, self.SEED_SIZE))  # Seed for the saved images

        # Build and train progressive growing model
        for i in range(self.NUM):
            # Load data
            training_data, y_real, y_fake = self.load_data(i + 2)  # Load data

            # Skip creating new models if the initial models are not trained
            if i != 0:
                discriminator = self.build_new_discriminator(discriminator, i+2)  # Build the new discriminator model
                discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])  # Compile the new discriminator model
                generator = self.build_new_generator(generator)  # Build the new generator model

                random_input = Input(shape=(self.SEED_SIZE, ))  # Create a random input
                generated_image = generator(random_input)  # Create generated image

                discriminator.trainable = False  # Prevent the weights to be adjusted
                validity = discriminator(generated_image)  # Create validity

                combined = Model(random_input, validity)  # Create a combine model
                combined.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])  # Compile a combine model

            # Start training
            for epoch in range(self.EPOCH_PER_NUM):
                idx = np.random.randint(0, training_data.shape[0], self.BATCH_SIZE)
                x_real = training_data[idx]

                # Generate some images
                seed = np.random.normal(0, 1, (self.BATCH_SIZE, self.SEED_SIZE))
                x_fake = generator.predict(seed)

                # Train discriminator on real and fake
                discriminator_metric_real = discriminator.train_on_batch(x=x_real, y=y_real)
                discriminator_metric_generated = discriminator.train_on_batch(x=x_fake, y=y_fake)
                discriminator_metric = 0.5 * np.add(discriminator_metric_real, discriminator_metric_generated)

                # Train the generator on calculate loss
                generator_metric = combined.train_on_batch(seed, y_real)

                # Save the image
                if epoch % self.SAVE_FREQ == 0:
                    self.save_images(fixed_seed, generator)
                    print(f"Image_size {2**(i+2)}, Epoch {epoch}, Discriminator accuracy: {discriminator_metric[1]}, Generator accuracy: {generator_metric[1]}")

        generator_path = "C:\\Users\\phuoc\\OneDrive\\Desktop\\CODE\\DeepLearning\\Model\\Advanced_Anime_Generator.h5"
        generator.save(generator_path)


# Clip model weights to a given hypercube for ProgressiveGrowingGan2
class ClipConstraint(object):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value

    # clip model weights to hypercube
    def __call__(self, weights):
        return K.clip(weights, -self.clip_value, self.clip_value)

    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}


class ProgressiveGrowingGan2(object):
    def __init__(self):
        self.IMAGE_CHANNEL = 3  # 1 for grayscale and 3 for RGB
        self.IMAGE_SHAPE = (4, 4, 3)  # Image shape
        self.NUM = 5  # Resolution of final images will be 2^(NUM+1)
        self.LAYERS = 128  # Layers for convolutional neural network
        self.LR = 0.00005  # Learning rate
        self.NUM_CRITIC = 5  # Number of critic model is trained before training generator

        self.PREVIEW_ROW = 3  # Number of images per row when preview images
        self.PREVIEW_COLUMN = 3  # Number of images per column when preview images
        self.PREVIEW_MARGIN = 16  # Space between images
        self.SAVE_FREQ = 100  # frequency of saving images during training

        self.SEED_SIZE = 100  # Size vector to generate images from

        self.EPOCH_PER_NUM = 100  # Number of epoch per resolution for training
        self.BATCH_SIZE = 64  # Size of image batches for training

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):
        input_latent = Input(shape=(self.SEED_SIZE,))
        initializer = RandomNormal(stddev=0.02)

        x = Dense(2 * 2 * 128, input_dim=self.SEED_SIZE)(input_latent)
        x = LeakyReLU(alpha=0.2)(x)
        x = Reshape((2, 2, 128))(x)

        x = UpSampling2D()(x)
        x = Conv2D(self.LAYERS, kernel_size=2, padding='same', kernel_initializer=initializer)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(self.IMAGE_CHANNEL, kernel_size=2, padding='same', kernel_initializer=initializer)(x)
        output_img = Activation('tanh')(x)

        generator = Model(input_latent, output_img)
        return generator

    def build_new_generator(self, generator):
        old_model = generator.layers[-3].output
        initializer = RandomNormal(stddev=0.02)

        x = UpSampling2D()(old_model)
        x = Conv2D(self.LAYERS, kernel_size=2, padding='same', kernel_initializer=initializer)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(self.IMAGE_CHANNEL, kernel_size=2, padding='same', kernel_initializer=initializer)(x)
        x = Activation('tanh')(x)

        new_generator = Model(generator.get_input_at(0), x)
        return new_generator

    def build_critic(self):
        input_image = Input(shape=self.IMAGE_SHAPE)
        initializer = RandomNormal(stddev=0.02)
        const = ClipConstraint(0.01)

        x = Conv2D(self.LAYERS, kernel_size=2, strides=2, padding='same', input_shape=self.IMAGE_SHAPE, kernel_initializer=initializer, kernel_constraint=const)(input_image)
        x = LeakyReLU(alpha=0.2)(x)

        x = Dropout(0.25)(x)
        x = Conv2D(self.LAYERS, kernel_size=2, strides=2, padding='same', kernel_initializer=initializer, kernel_constraint=const)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Flatten()(x)
        x = Dense(1, activation='linear')(x)

        discriminator = Model(input_image, x)
        return discriminator

    def build_new_critic(self, discriminator, old_shape):
        input_shape = (2**old_shape, 2**old_shape, 3)
        input_img = Input(shape=input_shape)
        initializer = RandomNormal(stddev=0.02)
        const = ClipConstraint(0.01)

        x = Conv2D(self.LAYERS, kernel_size=2, strides=2, padding='same', input_shape=input_shape, kernel_initializer=initializer, kernel_constraint=const)(input_img)
        x = LeakyReLU(alpha=0.2)(x)

        for i in range(3, len(discriminator.layers) - 2):
            x = discriminator.layers[i](x)

        x = Dropout(0.25)(x)
        x = Conv2D(self.LAYERS, kernel_size=2, strides=2, padding='same', kernel_initializer=initializer, kernel_constraint=const)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Flatten()(x)
        x = Dense(1, activation='linear')(x)

        new_discriminator = Model(input_img, x)
        return new_discriminator

    def build_gan(self, generator, critic):
        critic.trainable = False

        model = Sequential()
        model.add(generator)
        model.add(critic)

        model.compile(loss=self.wasserstein_loss, optimizer=RMSprop(lr=self.LR), metrics=['accuracy'])
        return model

    def save_images(self, noise, generator):
        save_path = "N:\\CODE\\DeepLearning\\Generated_Images_6\\"
        path = glob.glob(save_path + "*.png")

        generated_images = generator.predict(noise)
        generated_images = 0.5 * generated_images + 0.5

        shape = generated_images.shape[1]

        image_array = np.full((
            self.PREVIEW_MARGIN + (self.PREVIEW_ROW * (shape + self.PREVIEW_MARGIN)),
            self.PREVIEW_MARGIN + (self.PREVIEW_COLUMN * (shape + self.PREVIEW_MARGIN)), 3),
            255, dtype=np.uint8)

        image_count = 0
        for row in range(self.PREVIEW_ROW):
            for col in range(self.PREVIEW_COLUMN):
                r = row * (shape + 16) + self.PREVIEW_MARGIN
                c = col * (shape + 16) + self.PREVIEW_MARGIN
                image_array[r:r+shape, c:c+shape] = generated_images[image_count] * 255
                image_count += 1

        result = Image.fromarray(image_array)
        result.save(save_path + str(len(path)) + ".png")

    def load_data(self, num):
        data_path = glob.glob("N:\\CODE\\DeepLearning\\Anime_Images\\*.png")
        training_data = np.zeros((len(data_path), 2**num, 2**num, self.IMAGE_CHANNEL))

        for i in range(len(data_path)):
            training_data[i] = img_to_array(load_img(data_path[i], color_mode="rgb", target_size=(2**num, 2**num)))

        training_data = training_data.astype('float16')
        training_data = training_data / 127.5 - 1.0

        y_real = -np.ones((int(self.BATCH_SIZE/2), 1))
        y_fake = np.ones((int(self.BATCH_SIZE/2), 1))

        return training_data, y_real, y_fake

    def train(self):
        optimizer = RMSprop(lr=self.LR)  # The value is taken from research paper
        critic_loss_real = []
        critic_loss_fake = []
        generator_acc = []

        critic = self.build_critic()  # Build the discriminator model
        critic.compile(loss=self.wasserstein_loss, optimizer=optimizer, metrics=['accuracy'])  # Compile the critic model
        generator = self.build_generator()  # Build the generator model
        gan_model = self.build_gan(generator, critic)  # Build gan model

        fixed_seed = np.random.normal(0, 1, (self.PREVIEW_ROW * self.PREVIEW_COLUMN, self.SEED_SIZE))  # Seed for the saved images

        # Build and train progressive growing model
        for i in range(self.NUM):
            # Load data
            training_data, y_real, y_fake = self.load_data(i + 2)  # Load data

            # Skip creating new models if the initial models are not trained
            if i != 0:
                critic = self.build_new_critic(critic, i+2)  # Build the new discriminator model
                critic.compile(loss=self.wasserstein_loss, optimizer=optimizer, metrics=['accuracy'])  # Compile the new critic model
                generator = self.build_new_generator(generator)  # Build the new generator model
                gan_model = self.build_gan(generator, critic)  # Build gan model

            # Start training
            for epoch in range(self.EPOCH_PER_NUM):
                critic_loss_real_temp = []
                critic_loss_fake_temp = []

                for j in range(self.NUM_CRITIC):
                    idx = np.random.randint(0, training_data.shape[0], int(self.BATCH_SIZE/2))
                    x_real = training_data[idx]

                    # Generate some images
                    seed = np.random.normal(0, 1, (int(self.BATCH_SIZE/2), self.SEED_SIZE))
                    x_fake = generator.predict(seed)

                    # Train discriminator on real and fake
                    critic_metric_real = critic.train_on_batch(x=x_real, y=y_real)
                    critic_metric_generated = critic.train_on_batch(x=x_fake, y=y_fake)
                    critic_loss_real_temp.append(critic_metric_real)
                    critic_loss_fake_temp.append(critic_metric_generated)

                critic_loss_real.append(np.mean(critic_loss_real_temp))
                critic_loss_fake.append(np.mean(critic_loss_fake_temp))

                x_gan = np.random.normal(0, 1, (self.BATCH_SIZE, self.SEED_SIZE))
                y_gan = -np.ones((self.BATCH_SIZE, 1))

                generator_metric = gan_model.train_on_batch(x_gan, y_gan)  # Train the gan model
                generator_acc.append(generator_metric)

                # Save the image
                if epoch % self.SAVE_FREQ == 0:
                    self.save_images(fixed_seed, generator)
                    print(f"Image_size {2**(i+2)}, Epoch {epoch}, Critic loss real: {critic_loss_real[-1]},"
                          f" Critic loss fake: {critic_loss_fake[-1]}, Generator accuracy: {generator_acc[-1]}")

        generator_path = "C:\\Users\\phuoc\\OneDrive\\Desktop\\CODE\\DeepLearning\\Model\\Advanced_Anime_Generator.h5"
        generator.save(generator_path)


# Take randomly-weighted average of 2 tensors for ProgressiveGrowingGan3
class RandomWeightAverage(_Merge):
    def _merge_function(self, inputs):
        weights = K.random_uniform((64, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


class ProgressiveGrowingGan3(object):
    def __init__(self):
        self.IMAGE_CHANNEL = 3  # 1 for grayscale and 3 for RGB
        self.IMAGE_SHAPE = (4, 4, 3)  # Image shape
        self.NUM = 5  # Resolution of final images will be 2^(NUM+1)
        self.LAYERS = [16, 32, 64, 128, 256]  # Layers for convolutional neural network
        self.NUM_CRITIC = 5  # Number of critic model is trained before training generator
        self.GRADIENT_PENALTY_WEIGHT = 10  # Value taken from paper

        self.PREVIEW_ROW = 3  # Number of images per row when preview images
        self.PREVIEW_COLUMN = 3  # Number of images per column when preview images
        self.PREVIEW_MARGIN = 16  # Space between images
        self.SAVE_FREQ = 100  # frequency of saving images during training

        self.SEED_SIZE = 100  # Size vector to generate images from

        self.EPOCH_PER_NUM = [100, 200, 500, 1000, 2000]  # Number of epoch per resolution for training
        self.BATCH_SIZE = 64  # Size of image batches for training

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples, gradient_penalty_weight):
        gradients = K.gradients(y_pred, averaged_samples)[0]
        gradients_square = K.square(gradients)
        gradients_square_num = K.sum(gradients_square, axis=np.arange(1, len(gradients_square.shape)))
        gradients_l2_norm = K.sqrt(gradients_square_num)
        gradients_penalty = gradient_penalty_weight * K.square(1 - gradients_l2_norm)

        return K.mean(gradients_penalty)

    def build_generator(self):
        input_latent = Input(shape=(self.SEED_SIZE,))

        x = Dense(1 * 1 * self.LAYERS[len(self.LAYERS)-1], input_dim=self.SEED_SIZE)(input_latent)
        x = LeakyReLU(alpha=0.2)(x)
        x = Reshape((1, 1, self.LAYERS[len(self.LAYERS)-1]))(x)

        x = UpSampling2D()(x)
        x = Conv2D(self.LAYERS[len(self.LAYERS)-1], kernel_size=2, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = UpSampling2D()(x)
        x = Conv2D(self.LAYERS[len(self.LAYERS)-1], kernel_size=2, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(self.IMAGE_CHANNEL, kernel_size=2, padding='same', kernel_initializer='he_normal')(x)
        output_img = Activation('tanh')(x)

        generator = Model(input_latent, output_img)
        return generator

    def build_new_generator(self, generator, layer_index):
        old_model = generator.layers[-3].output

        x = UpSampling2D()(old_model)
        x = Conv2D(self.LAYERS[len(self.LAYERS) - 1 - layer_index], kernel_size=2, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(self.IMAGE_CHANNEL, kernel_size=2, padding='same', kernel_initializer='he_normal')(x)
        x = Activation('tanh')(x)

        new_generator = Model(generator.get_input_at(0), x)
        return new_generator

    def build_critic(self):
        input_image = Input(shape=self.IMAGE_SHAPE)

        x = Conv2D(self.LAYERS[0], kernel_size=2, strides=2, padding='same', input_shape=self.IMAGE_SHAPE, kernel_initializer='he_normal')(input_image)
        x = LeakyReLU(alpha=0.2)(x)

        x = Dropout(0.25)(x)
        x = Conv2D(self.LAYERS[0], kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Flatten()(x)
        x = Dense(1, activation='linear')(x)

        discriminator = Model(input_image, x)
        return discriminator

    def build_new_critic(self, discriminator, old_shape, layer_index):
        input_shape = (2 ** old_shape, 2 ** old_shape, 3)
        input_img = Input(shape=input_shape)

        x = Conv2D(self.LAYERS[0], kernel_size=2, strides=2, padding='same', input_shape=input_shape, kernel_initializer='he_normal')(input_img)
        x = LeakyReLU(alpha=0.2)(x)

        for i in range(3, len(discriminator.layers) - 2):
            x = discriminator.layers[i](x)

        x = Dropout(0.25)(x)
        x = Conv2D(self.LAYERS[layer_index], kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Flatten()(x)
        x = Dense(1, activation='linear')(x)

        new_discriminator = Model(input_img, x)
        return new_discriminator

    def build_model(self, mode, x_train, layer_index, old_generator=None, old_critic=None, shape=2):
        if mode == 0:
            generator = self.build_generator()
            critic = self.build_critic()
        else:
            generator = self.build_new_generator(old_generator, layer_index)
            critic = self.build_new_critic(old_critic, shape, layer_index)

        for layer in critic.layers:
            layer.trainable = False

        critic.trainable = False
        generator_input = Input(shape=(self.SEED_SIZE,))
        generator_layers = generator(generator_input)
        critic_layers_for_generator = critic(generator_layers)
        generator_model = Model(inputs=[generator_input], outputs=[critic_layers_for_generator])
        generator_model.compile(optimizer=Adam(lr=0.0001, beta_1=0.5, beta_2=0.9), loss=self.wasserstein_loss, metrics=['accuracy'])

        for layer in critic.layers:
            layer.trainable = True
        for layer in generator.layers:
            layer.trainable = False

        critic.trainable = True
        generator.trainable = False

        real_samples = Input(shape=x_train.shape[1:])
        generator_input_for_critic = Input(shape=(self.SEED_SIZE,))
        generated_samples_for_critic = generator(generator_input_for_critic)
        critic_output_from_generator = critic(generated_samples_for_critic)
        critic_output_from_real_samples = critic(real_samples)

        averaged_samples = RandomWeightAverage()([real_samples, generated_samples_for_critic])
        averaged_samples_out = critic(averaged_samples)

        partial_gp_loss = partial(self.gradient_penalty_loss, averaged_samples=averaged_samples, gradient_penalty_weight=self.GRADIENT_PENALTY_WEIGHT)
        partial_gp_loss.__name__ = 'gradient_penalty'

        critic_model = Model(inputs=[real_samples, generator_input_for_critic], outputs=[critic_output_from_real_samples, critic_output_from_generator, averaged_samples_out])
        critic_model.compile(optimizer=Adam(lr=0.0001, beta_1=0.5, beta_2=0.9), loss=[self.wasserstein_loss, self.wasserstein_loss, partial_gp_loss], metrics=['accuracy'])

        return critic_model, generator_model, critic, generator

    def save_images(self, noise, generator):
        save_path = "N:\\CODE\\DeepLearning\\Generated_Images_1\\"
        path = glob.glob(save_path + "*.png")

        generated_images = generator.predict(noise)
        generated_images = 0.5 * generated_images + 0.5

        shape = generated_images.shape[1]

        image_array = np.full((
            self.PREVIEW_MARGIN + (self.PREVIEW_ROW * (shape + self.PREVIEW_MARGIN)),
            self.PREVIEW_MARGIN + (self.PREVIEW_COLUMN * (shape + self.PREVIEW_MARGIN)), 3),
            255, dtype=np.uint8)

        image_count = 0
        for row in range(self.PREVIEW_ROW):
            for col in range(self.PREVIEW_COLUMN):
                r = row * (shape + 16) + self.PREVIEW_MARGIN
                c = col * (shape + 16) + self.PREVIEW_MARGIN
                image_array[r:r + shape, c:c + shape] = generated_images[image_count] * 255
                image_count += 1

        result = Image.fromarray(image_array)
        result.save(save_path + str(len(path)) + ".png")

    def load_data(self, num):
        data_path = glob.glob("N:\\CODE\\DeepLearning\\Anime_Images\\*.png")
        training_data = np.zeros((len(data_path), 2 ** num, 2 ** num, self.IMAGE_CHANNEL))

        for i in range(len(data_path)):
            training_data[i] = img_to_array(load_img(data_path[i], color_mode="rgb", target_size=(2 ** num, 2 ** num)))

        training_data = training_data.astype('float16')
        training_data = training_data / 127.5 - 1.0

        y_real = -np.ones((self.BATCH_SIZE, 1))
        y_fake = np.ones((self.BATCH_SIZE, 1))
        dummy = np.zeros((self.BATCH_SIZE, 1))

        return training_data, y_real, y_fake, dummy

    def train(self):
        critic_loss = []
        generator_loss = []

        fixed_seed = np.random.normal(0, 1, (self.PREVIEW_ROW * self.PREVIEW_COLUMN, self.SEED_SIZE))  # Seed for the saved images

        # Build and train progressive growing model
        for i in range(self.NUM):
            training_data, y_real, y_fake, dummy = self.load_data(i + 2)  # Load data

            # Skip creating new models if the initial models are not trained
            if i != 0:
                critic_model, generator_model, critic, generator = self.build_model(1, training_data, i, old_critic=critic, old_generator=generator, shape=(i+2))
            else:
                critic_model, generator_model, critic, generator = self.build_model(0, training_data, i)

            # Start training
            for epoch in range(self.EPOCH_PER_NUM[i]):
                mini_batch_size = self.BATCH_SIZE * self.NUM_CRITIC

                for j in range(int(training_data.shape[0]) // mini_batch_size):
                    critic_mini_batches = training_data[j * mini_batch_size: (j+1) * mini_batch_size]

                    for k in range(self.NUM_CRITIC):
                        img_batch = critic_mini_batches[k * self.BATCH_SIZE: (k+1) * self.BATCH_SIZE]
                        print(np.shape(img_batch))
                        input('wait')
                        noise = np.random.rand(self.BATCH_SIZE, self.SEED_SIZE).astype(np.float32)
                        critic_loss.append(critic_model.train_on_batch([img_batch, noise], [y_real, y_fake, dummy]))

                    generator_loss.append(generator_model.train_on_batch(np.random.rand(self.BATCH_SIZE, self.SEED_SIZE), y_real))

                # Save the image
                if epoch % self.SAVE_FREQ == 0:
                    self.save_images(fixed_seed, generator)
                    print('Image size: ' + str(2**(i+2)) + 'x' + str(2**(i+2)) + ' Critic loss: ' + str(critic_loss[-1]) + ' Generator loss: ' + str(generator_loss[-1]))

        generator_path = "C:\\Users\\phuoc\\OneDrive\\Desktop\\CODE\\DeepLearning\\Model\\Advanced_Anime_Generator.h5"
        generator.save(generator_path)


class Evolution_GAN(object):
    def __init__(self):
        # config = tf_version.ConfigProto()
        # config.gpu_options.allow_growth = True
        # sess = tf_version.Session(config=config)
        # tf_version.keras.backend.set_session(sess)

        self.img_size = 216  # Size of images
        self.img_channel = 3  # Channel of images. 1 for grayscale and 3 for rgb

        self.seed_size = self.img_size ** 2  # Size of the input latent for the generator
        self.preview_row = 1  # Number of saved images in a row
        self.preview_column = 1  # Number of saved images in a row
        self.preview_margin = 10  # Number of space between saved images
        self.save_freq = 100  # Frequency to save image

        self.mutation_rate = 0.05  # Rate of mutation which is used to introduce new random weights for the generator
        self.population = 10  # Total number of generators used for evolution

        self.learning_rate = 0.0005  # Rate of learning used for optimizer
        self.epoch = 50000  # Number of epoch
        self.batch_size = 3  # Size of batch
        self.evolve_freq = 1000  # Frequency for evolution
        self.layers = np.shape(self.build_generator().get_weights())[0]  # Number of layers which is used to get_weights() and set_weights()

        self.networks = []

    def build_generator(self):
        input_latent = Input(shape=(self.seed_size, ))

        x = Dense(1 * 1 * 512, input_dim=self.seed_size)(input_latent)
        x = LeakyReLU(alpha=0.2)(x)
        x = Reshape((1, 1, 512))(x)

        # Image_Size 3x3
        x = UpSampling2D(size=3)(x)
        x = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        # Image_Size 9x9
        x = UpSampling2D(size=3)(x)
        x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        # Image_Size 27x27
        x = UpSampling2D(size=3)(x)
        x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        # Image_Size 54x54
        x = UpSampling2D(size=2)(x)
        x = Conv2D(64, kernel_size=(2, 2), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        # Image_Size 108x108
        x = UpSampling2D(size=2)(x)
        x = Conv2D(32, kernel_size=(2, 2), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        # Image_Size 216x216
        x = UpSampling2D(size=2)(x)
        x = Conv2D(16, kernel_size=(2, 2), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(self.img_channel, kernel_size=(2, 2), strides=(1, 1), padding='same')(x)
        output_img = Activation('tanh')(x)

        return Model(inputs=input_latent, outputs=output_img)

    def build_critic(self):
        input_img = Input(shape=(self.img_size, self.img_size, self.img_channel))

        x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', input_shape=(self.img_size, self.img_size, self.img_channel))(input_img)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Dropout(0.2)(x)
        x = Flatten()(x)
        x = Dense(1, activation='sigmoid')(x)

        critic_model = Model(inputs=input_img, outputs=x)
        critic_model.compile(loss='binary_crossentropy', optimizer=Adam(1.5e-4, 0.5), metrics=['accuracy'])

        return critic_model

    def build_gan(self, critic, generator):
        critic.trainable = False

        model = Sequential()
        model.add(generator)
        model.add(critic)

        model.compile(loss='binary_crossentropy', optimizer=Adam(self.learning_rate, 0.5), metrics=['accuracy'])
        return model

    def breading(self, indexes):
        weight1 = self.networks[indexes[0]].get_weights()
        weight2 = self.networks[indexes[1]].get_weights()

        for i in range(self.layers):
            probability = random.random()  # Value used for breeding

            if probability <= self.mutation_rate:  # Introduce new weights to the neural networks
                shape = 1

                for j in range(len(np.shape(weight1[i]))):
                    shape *= np.shape(weight1[i])[j]

                sub_weight1 = np.random.randn(shape)
                sub_weight2 = np.random.randn(shape)

                weight1[i] = np.reshape(sub_weight1, np.shape(weight1[i]))
                weight2[i] = np.reshape(sub_weight2, np.shape(weight2[i]))
            elif self.mutation_rate < probability < (1 - self.mutation_rate) / 2.0:  # Copy the weights of parent1 to child1 and parent2 to child2
                weight1[i] = self.networks[indexes[0]].get_weights()[i]
                weight2[i] = self.networks[indexes[1]].get_weights()[i]
            else:  # Copy the weights of parent2 to child1 and parent1 to child2
                weight1[i] = self.networks[indexes[1]].get_weights()[i]
                weight2[i] = self.networks[indexes[0]].get_weights()[i]

        self.networks[indexes[0]].set_weights(weight1)  # Set the weights for child1
        self.networks[indexes[1]].set_weights(weight2)  # Set the weights for child2

    # Calculate fitness score based on the score from critic to the generator
    def fitness(self, critic):
        results = []
        values = []
        value = 0.0

        for generator in self.networks:
            seed = np.random.normal(0, 1, (1, self.seed_size))
            generated_img = generator.predict(seed)
            results.append(critic.predict(generated_img)[0][0])

        sum_result = np.sum(results)

        values.append(value)
        for i in range(len(self.networks)-1):
            value += results[i] / sum_result
            values.append(value)
        values.append(1.0)

        return values

    # Use genetic evolution to the population of generators
    def evolve(self, critic):
        parents_probability = self.fitness(critic)
        indexes = np.zeros((int(self.population / 2), 2))

        for i in range(len(indexes)):
            sub_index = []

            while len(sub_index) < 2:
                probability = np.random.rand()

                for j in range(1, len(parents_probability)):
                    if parents_probability[j-1] < probability < parents_probability[j]:
                        sub_index.append(j-1)

            indexes[i][0] = sub_index[0]
            indexes[i][1] = sub_index[1]

        multi_processing = mp.Pool(processes=10)
        multi_processing.imap(self.breading, indexes)

        multi_processing.close()
        multi_processing.join()

    def load_data(self):
        data_path = glob.glob("N:\\CODE\\DeepLearning\\Anime_Images\\*.png")
        training_data = []

        for index in range(len(data_path)):
            image = Image.open(data_path[index]).resize((self.img_size, self.img_size), Image.ANTIALIAS)
            training_data.append(np.asanyarray(image))

        training_data = np.reshape(training_data, (-1, self.img_size, self.img_size, self.img_channel))
        training_data = training_data / 127.5 - 1.0  # Make the image which is originally between 0 and 255 into between -1 and 1
        training_data = training_data.astype('float16')

        y_real = np.ones((self.batch_size, 1))
        y_real = y_real.astype('float16')

        y_fake = np.zeros((self.batch_size, 1))
        y_fake = y_fake.astype('float16')

        return training_data, y_real, y_fake

    def save_images(self, seed, generator):
        save_path = "N:\\CODE\\DeepLearning\\Generated_Images_7\\"
        path = glob.glob(save_path + "*.png")

        generated_images = generator.predict(seed)
        generated_images = 0.5 * generated_images + 0.5

        shape = generated_images.shape[1]

        image_array = np.full((
            self.preview_margin + (self.preview_row * (shape + self.preview_margin)),
            self.preview_margin + (self.preview_column * (shape + self.preview_margin)), 3),
            255, dtype=np.uint8)

        image_count = 0
        for row in range(self.preview_row):
            for col in range(self.preview_column):
                r = row * (shape + 16) + self.preview_margin
                c = col * (shape + 16) + self.preview_margin
                image_array[r:r+shape, c:c+shape] = generated_images[image_count] * 255
                image_count += 1

        result = Image.fromarray(image_array)
        result.save(save_path + str(len(path)) + ".png")

    def evaluate_generator(self, critic, seed):
        results = []

        for generator in self.networks:
            generated_img = generator.predict(seed)
            results.append(critic.predict(generated_img)[0][0])

        return np.argmax(results)

    def train(self):
        critic = self.build_critic()
        gan_model = []
        seeds = []
        fixed_seed = np.random.normal(0, 1, (self.preview_row * self.preview_column, self.seed_size))  # Seed for the saved images
        index = 1

        # Initialize the population
        for i in range(self.population):
            self.networks.append(self.build_generator())
            gan_model.append(self.build_gan(critic, self.networks[i]))

        training_data, y_real, y_fake = self.load_data()  # Load data

        # Start training
        for i in range(self.epoch):
            print('epoch ' + str(i))
            seeds.clear()

            for j in range(self.population):
                print(training_data.shape[0])
                idx = np.random.randint(0, training_data.shape[0], self.batch_size)
                x_real = training_data[idx]

                seed = np.random.normal(0, 1, (self.batch_size, self.seed_size))
                seeds.append(seed)
                x_fake = self.networks[j].predict(seed)

                critic.train_on_batch(x=x_real, y=y_real)
                critic.train_on_batch(x=x_fake, y=y_fake)

            for j in range(self.population):
                gan_model[j].train_on_batch(seeds[j], y_real)

            if i % self.evolve_freq == 0 and i != 0:
                print('Start evolution process')
                self.evolve_freq += 900 * index
                index += 1
                self.evolve(critic)

            if i % self.save_freq == 0:
                index = self.evaluate_generator(critic, fixed_seed)
                self.save_images(fixed_seed, self.networks[int(index)])

        index = self.evaluate_generator(critic, fixed_seed)
        self.networks[int(index)].save("C:\\Users\\phuoc\\OneDrive\\Desktop\\CODE\\DeepLearning\\Model\\Evolution_GAN.h5")


if __name__ == '__main__':
    # dc_gan = DC_GAN()
    # w_gan = W_GAN()
    # advance_gan = ProgressiveGrowingGan()
    # advance_gan.run()
    # w_gan.train()
    # dc_gan.train()
    # frame_to_video()
    # advance_gan = ProgressiveGrowingGan3()
    # advance_gan.train()
    egan = Evolution_GAN()
    # print(egan.build_generator().summary())
    egan.train()
    print('done')
