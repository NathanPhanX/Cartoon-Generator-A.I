from __future__ import print_function, division
from builtins import range, input
from datetime import datetime
from keras.layers import Input, Lambda, Dense, Flatten, AveragePooling2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Model, Sequential
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from scipy.optimize import fmin_l_bfgs_b

import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import PIL


def VGG16_AvgPool(shape):
    vgg = VGG16(input_shape=shape, weights='imagenet', include_top=False)
    new_model = Sequential()

    for layer in vgg.layers:
        if layer.__class__ == MaxPooling2D:
            new_model.add(AveragePooling2D())
        else:
            new_model.add(layer)

    return new_model


def VGG16_AvgPool_CutOff(shape, num_convs):
    if num_convs < 1 or num_convs > 13:
        print("Number of convolution layers must be between 1 and 13")
        return None

    model = VGG16_AvgPool(shape)
    new_model = Sequential()
    num = 0

    for layer in model.layers:
        if layer.__class__ == Conv2D:
            num += 1

        new_model.add(layer)

        if num >= num_convs:
            break

    return new_model


def unpreprocess(img):
    img[..., 0] += 103.939
    img[..., 1] += 116.779
    img[..., 0] += 126.68
    img = img[..., ::-1]
    return img


def scale_img(x):
    x = x - x.min()
    x = x / x.max()
    return x


def img_style():
    # Load the image
    path = "C:\\Users\\phuoc\\OneDrive\\Desktop\\CODE\\DeepLearning\\Data\\test1.jpg"
    img = image.load_img(path)

    # Convert image to array and preprocess for vgg
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    batch_shape = x.shape
    shape = x.shape[1:]

    # Make the content model
    content_model = VGG16_AvgPool_CutOff(shape, 11)

    # Make target
    target = K.variable(content_model.predict(x))

    # Define loss in keras
    loss = K.mean(K.square(target - content_model.output))

    # Define gradients which are needed by the optimizer
    grads = K.gradients(loss, content_model.input)

    # Get loss and gradients
    get_loss_and_grads = K.function(inputs=[content_model.input], outputs=[loss] + grads)

    def get_loss_and_grads_wrapper(x_vec):
        l, g = get_loss_and_grads([x_vec.reshape(*batch_shape)])
        return l.astype(np.float64), g.flatten().astype(np.float64)

    t0 = datetime.now()
    losses = []
    x = np.random.randn(np.prod(batch_shape))

    for i in range(10):
        x, l, _ = fmin_l_bfgs_b(func=get_loss_and_grads_wrapper, x0=x, maxfun=20)
        x = np.clip(x, -127, 127)
        print("iter=%s, loss=%s" % (i, l))
        losses.append(l)

    print("duration:", datetime.now() - t0)
    plt.plot(losses)
    plt.show()

    new_img = x.reshape(*batch_shape)
    final_img = unpreprocess(new_img)

    plt.imshow(scale_img(final_img[0]))
    plt.show()


def gram_matrix(img):
    # Input is (height, width, C) where C is feature maps
    # First, convert it to (C, H*W)
    X = K.batch_flatten(K.permute_dimensions(img, (2, 0, 1)))

    # Calculate gram matrix
    G = K.dot(X, K.transpose(X)) / img.get_shape().num_elements()

    return G


def style_loss(y, t):
    return K.mean(K.square(gram_matrix(y) - gram_matrix(t)))


def minimize(fn, epochs, batch_shape):
    t0 = datetime.now()
    losses = []
    x = np.random.randn(np.prod(batch_shape))

    for i in range(epochs):
        x, l, _ = fmin_l_bfgs_b(func=fn, x0=x, maxfun=20)
        x = np.clip(x, -127, 127)
        print("iter=%s, loss=%s" % (i, l))
        losses.append(l)

    print("duration:", datetime.now() - t0)
    plt.plot(losses)
    plt.show()

    new_img = x.reshape(*batch_shape)
    final_img = unpreprocess(new_img)
    return final_img[0]


def content_style():
    # Load the image
    path = "C:\\Users\\phuoc\\OneDrive\\Desktop\\CODE\\DeepLearning\\Data\\test1.jpg"
    img = image.load_img(path)

    # Convert image to array and preprocess for vgg
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    batch_shape = x.shape
    shape = x.shape[1:]

    vgg = VGG16_AvgPool(shape)
    symbolic_conv_output = [
        layer.get_output_at(1) for layer in vgg.layers \
        if layer.name.endswith('conv1')]

    multi_output_model = Model(vgg.input, symbolic_conv_output)

    style_layers_output = [K.variable(y) for y in multi_output_model.predict(x)]

    loss = 0

    for symbolic, actual in zip(symbolic_conv_output, style_layers_output):
        loss += style_loss(symbolic[0], actual[0])

    grads = K.gradients(loss, multi_output_model.input)

    get_loss_and_grads = K.function(inputs=[multi_output_model.input], outputs=[loss] + grads)

    def get_loss_and_grads_wrapper(x_vec):
        l, g = get_loss_and_grads([x_vec.reshape(*batch_shape)])
        return l.astype(np.float64), g.flatten().astype(np.float64)

    final_img = minimize(get_loss_and_grads_wrapper, 10, batch_shape)
    plt.imshow(scale_img(final_img))
    plt.show()


def load_img_and_preprocess(path, shape=None):
    # Load image
    img = image.load_img(path, target_size=shape)

    # Preprocess image for vgg
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return x


content_img = load_img_and_preprocess("C:\\Users\\phuoc\\OneDrive\\Desktop\\CODE\\DeepLearning\\Data\\test1.jpg")
h, w = content_img.shape[1:3]
style_img = load_img_and_preprocess("C:\\Users\\phuoc\\OneDrive\\Desktop\\CODE\\DeepLearning\\Data\\test1.jpg", (h,w))

batch_shape = content_img.shape
shape = content_img.shape[1:]

content_model = Model(vgg.input, vgg.layers[13].get_output_at(1))
content_target = K.variable(content_model.predict(content_img))

symbolic_conv_outputs = [layer.get_output_at(1) for layer in vgg.layers \
                         if layer.name.endswith('conv1')]

style_model = Model(vgg.input, symbolic_conv_outputs)
style_layers_outputs = [K.variable(y) for y in style_model.predict(style_img)]

style_weights = [1, 2, 3, 4, 5]

loss = K.mean(K.square(content_model.output - content_target))

for w, symbolic, actual in zip(style_weights, symbolic_conv_outputs, style_layers_outputs):
    loss += w * style_loss(symbolic[0], actual[0])

grads = K.gradients(loss, vgg.input)

get_loss_and_grads = K.function(inputs=[vgg.input], outputs=[loss] + grads)


def get_loss_and_grads_wrapper(x_vec):
    l, g = get_loss_and_grads([x_vec.reshape(*batch_shape)])
    return l.astype(np.float64), g.flatten().astype(np.float64)


final_img = minimize(get_loss_and_grads_wrapper, 10, batch_shape)
plt.imshow(scale_img(final_img))
plt.show()

