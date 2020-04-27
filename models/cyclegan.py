import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
import numpy as np
import PIL
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def load_img(path_to_img):
    img = image.load_img(path_to_img, target_size=(256, 256))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return tf.convert_to_tensor(img.astype('float32')/255)
    # img = tf.io.read_file(path_to_img)
    # img = tf.image.decode_image(img, channels=3)
    # img = tf.image.convert_image_dtype(img, tf.float32)
    # shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    # long_dim = tf.reduce_max(shape)
    # scale = max_dim / long_dim
    # new_shape = tf.cast(shape * scale, tf.int32)
    # img = tf.image.resize(img, new_shape)
    # img = img[tf.newaxis, :]
    # return img

def produceResult(baseFile, textureFile):
    content_image = load_img(baseFile)
    style_image = load_img(textureFile)
    # gan_layer = hub.KerasLayer('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1',
    #  input_shape=[256, 256, 3], trainable=False)
    #
    # model = keras.Sequential(layers=[gan_layer])
    # stylized_image = model.predict(content_image, style_image)[0]

    hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
    #stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
    stylized_image = hub_module(content_image, style_image)[0]
    img = tensor_to_image(stylized_image)
    img.save("result.jpg")


#produceResult('base.jpg', 'texture.jpg')
