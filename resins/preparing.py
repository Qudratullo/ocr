'''
 * @author [Zizhao Zhang]
 * @email [zizhao@cise.ufl.edu]
 * @create date 2017-05-25 02:21:13
 * @modify date 2017-05-25 02:21:13
 * @desc [description]
'''
import tensorflow as tf
import keras
# try:
#     from tensorflow.contrib import keras as keras
#     print ('load keras from tensorflow package')
# except:
#     print ('update your tensorflow')
from keras import models, layers


class UNet:
    def __init__(self):
        print('build UNet ...')

    def down_block(self, x, filters, kernel_size=(3, 3), padding="same", strides=1):
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
        p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
        return c, p

    def up_block(self, x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
        us = keras.layers.UpSampling2D((2, 2))(x)
        concat = keras.layers.Concatenate()([us, skip])
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
        return c

    def bottleneck(self, x, filters, kernel_size=(3, 3), padding="same", strides=1):
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
        return c

    def create_model(self, img_shape):
        f = [16, 32, 64, 128, 256]
        inputs = keras.layers.Input(img_shape)

        p0 = inputs
        c1, p1 = self.down_block(p0, f[0])  # 128 -> 64
        c2, p2 = self.down_block(p1, f[1])  # 64 -> 32
        c3, p3 = self.down_block(p2, f[2])  # 32 -> 16
        c4, p4 = self.down_block(p3, f[3])  # 16->8

        bn = self.bottleneck(p4, f[4])

        u1 = self.up_block(bn, c4, f[3])  # 8 -> 16
        u2 = self.up_block(u1, c3, f[2])  # 16 -> 32
        u3 = self.up_block(u2, c2, f[1])  # 32 -> 64
        u4 = self.up_block(u3, c1, f[0])  # 64 -> 128

        outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)
        model = keras.models.Model(inputs, outputs)

        return model

    # def get_crop_shape(self, target, refer):
    #     # width, the 3rd dimension
    #     cw = (target.get_shape()[2] - refer.get_shape()[2]).value
    #     assert (cw >= 0)
    #     if cw % 2 != 0:
    #         cw1, cw2 = int(cw / 2), int(cw / 2) + 1
    #     else:
    #         cw1, cw2 = int(cw / 2), int(cw / 2)
    #     # height, the 2nd dimension
    #     ch = (target.get_shape()[1] - refer.get_shape()[1]).value
    #     assert (ch >= 0)
    #     if ch % 2 != 0:
    #         ch1, ch2 = int(ch / 2), int(ch / 2) + 1
    #     else:
    #         ch1, ch2 = int(ch / 2), int(ch / 2)
    #
    #     return (ch1, ch2), (cw1, cw2)

    # def create_model(self, img_shape, num_class):
    #
    #     concat_axis = 3
    #     inputs = layers.Input(shape=img_shape)
    #
    #     conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1')(inputs)
    #     conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    #     pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    #     conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    #     conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    #     pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    #
    #     conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    #     conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    #     pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    #
    #     conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    #     conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    #     pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    #
    #     conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    #     conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    #
    #     up_conv5 = layers.UpSampling2D(size=(2, 2))(conv5)
    #     ch, cw = self.get_crop_shape(conv4, up_conv5)
    #     crop_conv4 = layers.Cropping2D(cropping=(ch, cw))(conv4)
    #     up6 = layers.concatenate([up_conv5, crop_conv4], axis=concat_axis)
    #     conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    #     conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    #
    #     up_conv6 = layers.UpSampling2D(size=(2, 2))(conv6)
    #     ch, cw = self.get_crop_shape(conv3, up_conv6)
    #     crop_conv3 = layers.Cropping2D(cropping=(ch, cw))(conv3)
    #     up7 = layers.concatenate([up_conv6, crop_conv3], axis=concat_axis)
    #     conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    #     conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    #
    #     up_conv7 = layers.UpSampling2D(size=(2, 2))(conv7)
    #     ch, cw = self.get_crop_shape(conv2, up_conv7)
    #     crop_conv2 = layers.Cropping2D(cropping=(ch, cw))(conv2)
    #     up8 = layers.concatenate([up_conv7, crop_conv2], axis=concat_axis)
    #     conv8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    #     conv8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    #
    #     up_conv8 = layers.UpSampling2D(size=(2, 2))(conv8)
    #     ch, cw = self.get_crop_shape(conv1, up_conv8)
    #     crop_conv1 = layers.Cropping2D(cropping=(ch, cw))(conv1)
    #     up9 = layers.concatenate([up_conv8, crop_conv1], axis=concat_axis)
    #     conv9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    #     conv9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    #
    #     ch, cw = self.get_crop_shape(inputs, conv9)
    #     conv9 = layers.ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(conv9)
    #     conv10 = layers.Conv2D(num_class, (1, 1))(conv9)
    #
    #     model = models.Model(inputs=inputs, outputs=conv10)
    #
    #     return model
