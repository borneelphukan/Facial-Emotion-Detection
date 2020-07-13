from keras import layers
from keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization
from keras.models import Sequential
from keras.layers import SeparableConv2D, GlobalAveragePooling2D, Input
from keras.models import Model
from keras.regularizers import l2

def xceptionNetwork(input_shape, num_classes, l2_regularization=0.01):
    reg = l2(l2_regularization)

    image = Input(input_shape)
    x = Convolution2D(8, (3, 3), strides=(1, 1), kernel_regularizer=reg, use_bias=False)(image)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(8, (3, 3), strides=(1, 1), kernel_regularizer=reg, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    res = Convolution2D(16, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    res = BatchNormalization()(res)

    x = SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=reg, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=reg, use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, res])

    res = Convolution2D(32, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    res = BatchNormalization()(res)

    x = SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=reg, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=reg, use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, res])

    res = Convolution2D(64, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    res = BatchNormalization()(res)

    x = SeparableConv2D(64, (3, 3), padding='same', kernel_regularizer=reg, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(64, (3, 3), padding='same', kernel_regularizer=reg, use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, res])

    res = Convolution2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    res = BatchNormalization()(res)

    x = SeparableConv2D(128, (3, 3), padding='same', kernel_regularizer=reg, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', kernel_regularizer=reg, use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, res])

    x = Convolution2D(num_classes, (3, 3), padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax',name='predictions')(x)

    model = Model(image, output)
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model

if __name__ == "__main__":
	input_shape = (64, 64, 1)
	num_classes = 7
	model = xceptionNetwork(input_shape, num_classes)