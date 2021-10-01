from __future__ import division

import tensorflow
import keras
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model, Model


def Class_Activation_Map(model_path, img_path, color_mode, target_size, last_activation_layer, last_dense_layer, scale):
    img = np.zeros((1, target_size[0], target_size[1], 1))  # Training data
    classifier = load_model(model_path)
    print(classifier.summary())

    img2 = load_img(path=img_path, color_mode='rgb', target_size=target_size)
    img1 = img_to_array(load_img(path=img_path, color_mode=color_mode, target_size=target_size))
    img1 = img1.astype('float32')
    img1 = img1 / 255.0
    img[0] = img1
    prediction = np.argmax(classifier.predict(img), axis=1)
    print(prediction)

    activation_layer = classifier.get_layer(last_activation_layer)
    model = Model(input=classifier.input, outputs=activation_layer.output)

    dense_layer = classifier.get_layer(last_dense_layer)
    weight = dense_layer.get_weights()[0]
    feature_map = model.predict(img)[0]

    new_weight = weight[:, prediction]

    print(feature_map.shape)
    print(new_weight.shape)

    N = len(new_weight)
    cam = np.zeros(feature_map.shape[:-1])
    for i in range(N):
        cam += feature_map[:, :, i] * new_weight[i]

    cam = sp.ndimage.zoom(cam, scale, order=1)

    print('print')
    plt.subplot(1, 2, 1)
    plt.imshow(img2, alpha=0.8)
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.title('Class_Activation_Map')
    plt.show()
