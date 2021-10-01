from __future__ import division

import os
import tensorflow
import keras
import numpy as np
import matplotlib.pyplot as plt
import Class_Activation_Map as heat_map

from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten, AveragePooling2D, MaxPooling2D, Conv2D, Activation, Concatenate, Dropout, BatchNormalization
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def identity_block(x, filters, kernel_size):
    x_shortcut = x

    f1, f2, f3 = filters

    x = Conv2D(filters=f1, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    # x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=f2, kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same')(x)
    # x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    # x = BatchNormalization(axis=3)(x)

    print(x.shape)
    print(x_shortcut.shape)

    x = Concatenate(axis=3)([x, x_shortcut])
    x = BatchNormalization(axis=3)(x, training=False)
    x = Activation('relu')(x)

    return x


def convolution_block(x, filters, kernel_size):
    x_shortcut = x

    f1, f2, f3, f4 = filters

    x = Conv2D(filters=f1, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    # x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=f2, kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same')(x)
    # x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    # x = BatchNormalization(axis=3)(x)

    x_shortcut = Conv2D(filters=f4, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x_shortcut)
    # x_shortcut = BatchNormalization(axis=3)(x_shortcut)

    x = Concatenate(axis=3)([x, x_shortcut])
    x = BatchNormalization(axis=3)(x, training=False)
    x = Activation('relu')(x)

    return x


def stem(x, regularizer):
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='valid', kernel_regularizer=l2(regularizer))(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', kernel_regularizer=l2(regularizer))(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(regularizer))(x)

    x1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)

    x2 = Conv2D(filters=96, kernel_size=(3, 3), strides=(2, 2), padding='valid', kernel_regularizer=l2(regularizer))(x)

    x = Concatenate(axis=3)([x1, x2])
    print('Checkpoint 1')

    x3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(regularizer))(x)
    x3 = Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding='valid', kernel_regularizer=l2(regularizer))(x3)

    x4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(regularizer))(x)
    x4 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(regularizer))(x4)
    x4 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', kernel_regularizer=l2(regularizer))(x4)
    x4 = Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding='valid', kernel_regularizer=l2(regularizer))(x4)

    x = Concatenate(axis=3)([x3, x4])
    print('Checkpoint 2')

    x5 = Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='valid', kernel_regularizer=l2(regularizer))(x)

    x6 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='valid')(x)

    x = Concatenate(axis=3)([x5, x6])
    x = BatchNormalization(axis=3)(x, training=False)
    x = Activation('relu')(x)
    print('Checkpoint 3')

    return x


def inception_A(x, regularizer):
    x_shortcut = x

    x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(regularizer))(x)

    x2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(regularizer))(x)
    x2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(regularizer))(x2)

    x3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(regularizer))(x)
    x3 = Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(regularizer))(x3)
    x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(regularizer))(x3)

    x = Concatenate(axis=3)([x1, x2, x3])
    x = Conv2D(filters=384, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(regularizer))(x)

    x = Concatenate(axis=3)([x, x_shortcut])
    x = BatchNormalization(axis=3)(x, training=False)
    x = Activation('relu')(x)

    return x


def inception_B(x, regularizer):
    x_shortcut = x

    x1 = Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(regularizer))(x)

    x2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(regularizer))(x)
    x2 = Conv2D(filters=160, kernel_size=(1, 7), strides=(1, 1), padding='same', kernel_regularizer=l2(regularizer))(x2)
    x2 = Conv2D(filters=192, kernel_size=(7, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(regularizer))(x2)

    x = Concatenate(axis=3)([x1, x2])
    x = Conv2D(filters=1154, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(regularizer))(x)

    x = Concatenate(axis=3)([x, x_shortcut])
    x = BatchNormalization(axis=3)(x, training=False)
    x = Activation('relu')(x)

    return x


def inception_C(x, regularizer):
    x_shortcut = x

    x1 = Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(regularizer))(x)

    x2 = Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(regularizer))(x)
    x2 = Conv2D(filters=224, kernel_size=(1, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(regularizer))(x2)
    x2 = Conv2D(filters=256, kernel_size=(3, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(regularizer))(x2)

    x = Concatenate(axis=3)([x1, x2])
    x = Conv2D(filters=2048, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(regularizer))(x)

    x = Concatenate(axis=3)([x, x_shortcut])
    x = BatchNormalization(axis=3)(x, training=False)
    x = Activation('relu')(x)

    return x


def reduction_A(x, regularizer):
    x1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)

    x2 = Conv2D(filters=384, kernel_size=(3, 3), strides=(2, 2), padding='valid', kernel_regularizer=l2(regularizer))(x)

    x3 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(regularizer))(x)
    x3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(regularizer))(x3)
    x3 = Conv2D(filters=384, kernel_size=(3, 3), strides=(2, 2), padding='valid', kernel_regularizer=l2(regularizer))(x3)

    x = Concatenate(axis=3)([x1, x2, x3])
    x = BatchNormalization(axis=3)(x, training=False)
    x = Activation('relu')(x)

    return x


def reduction_B(x, regularizer):
    x1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)

    x2 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(regularizer))(x)
    x2 = Conv2D(filters=384, kernel_size=(3, 3), strides=(2, 2), padding='valid', kernel_regularizer=l2(regularizer))(x2)

    x3 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(regularizer))(x)
    x3 = Conv2D(filters=288, kernel_size=(3, 3), strides=(2, 2), padding='valid', kernel_regularizer=l2(regularizer))(x3)

    x4 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(regularizer))(x)
    x4 = Conv2D(filters=288, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(regularizer))(x4)
    x4 = Conv2D(filters=320, kernel_size=(3, 3), strides=(2, 2), padding='valid', kernel_regularizer=l2(regularizer))(x4)

    x = Concatenate(axis=3)([x1, x2, x3, x4])
    x = BatchNormalization(axis=3)(x, training=False)
    x = Activation('relu')(x)

    return x


def inception_resnet_v2(input_layer, num_class, regularizer):
    x = stem(input_layer, regularizer)

    for i in range(5):
        x = inception_A(x, regularizer)

    print('Checkpoint 4')
    x = reduction_A(x, regularizer)
    print('Checkpoint 5')

    for i in range(10):
        x = inception_B(x, regularizer)

    print('Checkpoint 6')
    x = reduction_B(x, regularizer)
    print('Checkpoint 7')

    for i in range(5):
        x = inception_C(x, regularizer)
    print('Checkpoint 8')

    x = AveragePooling2D(pool_size=(2, 2), padding='valid')(x)
    x = Dropout(rate=0.2)(x)

    x = Flatten()(x)

    x = Dense(units=2048, activation='relu', kernel_regularizer=l2(regularizer))(x)
    x = Dense(units=1024, activation='relu', kernel_regularizer=l2(regularizer))(x)
    x = Dense(units=512, activation='relu', kernel_regularizer=l2(regularizer))(x)
    x = Dense(num_class, activation='softmax')(x)

    print('done')
    return x


def build_model(height, width, channel, num_class, regularizer):
    input_shape = (height, width, channel)

    x_input = Input(input_shape)
    # x = ZeroPadding2D(padding=(3, 3), data_format='channels_last')(x_input)
    x = inception_resnet_v2(x_input, num_class, regularizer)

    model = Model(inputs=x_input, outputs=x)

    return model


def preprocess_img(height, width, total, validation_split):
    path = "C:\\Users\\phuoc\\OneDrive\\Desktop\\CODE\\DeepLearning\\Data\\DroneControl\\"  # Path of the data located
    train_num = int(total * (1 - validation_split))  # Number of training data
    test_num = int(total * validation_split)  # Number of testing data
    x_train = np.zeros((train_num, height, width, 1))  # Training data
    y_train = np.zeros((train_num, ))  # Label of training data
    x_test = np.zeros((test_num, height, width, 1))  # Testing data
    y_test = np.zeros((test_num, ))  # Label of testing data
    img_type = ""  # Type of image
    index_img = 0  # Image index
    index1 = 0  # Index to determine image type
    index2 = 0  # Index of data which are training and testing
    check = False  # Check if the training data is finished loading

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    print('train_num ' + str(train_num))
    print('test_num ' + str(test_num))

    for i in range(total):
        # Reset the image type index if all the types of image is for a specific index2 is loaded
        if index1 > 9:
            index1 = 0

        # Increase the image type index if all the types of image is for a specific index2 is loaded
        if i % 10 == 0:
            index_img += 1

        # Reset the index of data and indicate training data is finished loading
        if index2 >= train_num:
            check = True
            index2 = 0

        # Determine the image type
        if index1 == 0:
            img_type = "up"
        elif index1 == 1:
            img_type = "down"
        elif index1 == 2:
            img_type = "left"
        elif index1 == 3:
            img_type = "right"
        elif index1 == 4:
            img_type = "forward"
        elif index1 == 5:
            img_type = "backward"
        elif index1 == 6:
            img_type = "stop"
        elif index1 == 7:
            img_type = "turn"
        elif index1 == 8:
            img_type = "nothing"
        elif index1 == 9:
            img_type = "land"

        direction = path + img_type + "\\" + img_type + str(index_img) + ".png"

        if check is not True:
            x_train[index2] = img_to_array(load_img(direction, color_mode="grayscale", target_size=(height, width)))
            y_train[index2] = index1
        else:
            x_test[index2] = img_to_array(load_img(direction, color_mode="grayscale", target_size=(height, width)))
            y_test[index2] = index1

        index1 += 1
        index2 += 1

    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')
    x_test = x_test.astype('float32')
    y_test = y_test.astype('float32')

    x_train = x_train / 255.0
    x_test = x_test / 255.0
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    print('Data loaded')
    return x_train, y_train, x_test, y_test


def main():
    print('\nStarting...')

    height = 90  # Height of the image
    width = 120  # Width of the image
    channel = 1  # Color mode of the image. 1 for grayscale and 3 for rgb
    num_class = 10  # Total different types of image for classification
    batch_size = 10  # Size of batch
    epoch = 50  # Number of epoch the model will train
    regularizer = 0.000001  # Regularize the kernel_regularizer
    total_data = 10000  # Total images
    validation_split = 0.2  # Ratio for splitting the validation image
    steps_per_epoch = 1000  # Number of training steps repeated per epoch
    validation_steps = 200  # Number of validation steps repeated per epoch
    lr = 0.01  # Learning rate
    decay = lr / (steps_per_epoch * epoch)  # Decay for learning rate
    path = "C:\\Users\\phuoc\\OneDrive\\Desktop\\CODE\\DeepLearning\\Model\\DroneControl.h5"  # Path of the saved model

    model = build_model(height, width, channel, num_class, regularizer)  # Build model
    x_train, y_train, x_test, y_test = preprocess_img(height, width, total_data, validation_split)  # Get the data

    training_gen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, featurewise_center=True, zca_whitening=True, channel_shift_range=0.5)  # Image generator for training
    validation_gen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, featurewise_center=True, zca_whitening=True, channel_shift_range=0.5)  # Image generator for testing

    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=lr, momentum=0.0, decay=decay, nesterov=False), metrics=['accuracy'])
    # r = model.fit(x=data, y=label, batch_size=batch_size, epochs=epoch, validation_split=0.15)
    r = model.fit_generator(generator=training_gen.flow(x=x_train, y=y_train, batch_size=batch_size), steps_per_epoch=steps_per_epoch,
                            validation_data=validation_gen.flow(x=x_test, y=y_test, batch_size=batch_size), validation_steps=validation_steps, max_queue_size=30,
                            callbacks=[ModelCheckpoint(filepath=path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')], workers=batch_size, epochs=epoch)
    # model.save(filepath=path)

    print(model.summary())

    # print('Evaluating...')
    # score = model.evaluate(x=data, y=label, batch_size=batch_size)
    # print('Test Lost: ', score[0])
    # print('Test Accuracy: ', score[1])

    plt.plot(r.history['loss'], label='loss')
    plt.plot(r.history['val_loss'], label='val_loss')
    plt.legend()
    plt.show()

    plt.plot(r.history['acc'], label='acc')
    plt.plot(r.history['val_acc'], label='val_acc')
    plt.legend()
    plt.show()


def predict():
    model_path = "C:\\Users\\phuoc\\OneDrive\\Desktop\\CODE\\DeepLearning\\Model\\DroneControl.h5"
    height = 90
    width = 120
    total = 10000
    result = 0

    classifier = load_model(filepath=model_path)
    print(classifier.summary())
    img, label, a, b = preprocess_img(height, width, total, 0)
    prediction = classifier.predict(img)

    test = np.argmax(label, axis=1)
    predict_result = np.argmax(prediction, axis=1)

    for i in range(total):
        if predict_result[i] == test[i]:
            result += 1

    print(result)
    print(str(int((result * 100) / total)) + '%')


def heat_map_classification(model_path, img_path, color_mode, target_size, last_activation_layer, last_dense_layer, scale):
    heat_map.Class_Activation_Map(model_path, img_path, color_mode, target_size, last_activation_layer, last_dense_layer, scale)


test = np.zeros((3, ))
test[0] = 450
test[1] = 700
test[2] = 100
label = to_categorical(test)

for i in label:
    print(i)

