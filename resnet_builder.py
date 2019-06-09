import tensorflow.keras.layers as layers
from tensorflow.keras.models import Sequential, Model


def identity_block(x, filter_size, filters):
    filters1, filters2, filters3 = filters
    
    x_shortcut = x
    
    x = layers.Conv2D(filters1, (1, 1), activation="relu")(x)
    
    x = layers.Conv2D(filters2, filter_size,
                      padding='same', activation="relu")(x)
    
    x = layers.Conv2D(filters3, (1, 1), activation="relu")(x)
    
    x = layers.add([x, x_shortcut])
    x = layers.Activation('relu')(x)
    return x


def conv_block(x, filter_size, filters):
    filters1, filters2, filters3 = filters
    
    x_shortcut = x
    x = layers.Conv2D(filters1, (1, 1), activation="relu")(x)
    
    x = layers.Conv2D(filters2, filter_size, strides=(2, 2), padding='same', activation="relu")(x)
    
    x = layers.Conv2D(filters3, (1, 1), activation="relu")(x)
    
    x_shortcut = layers.Conv2D(filters3, (1, 1), strides=(2, 2))(x_shortcut)
    
    x = layers.add([x, x_shortcut])
    x = layers.Activation("relu")(x)
    return x


def resnet50(input_shape, num_classes):
    x_input = layers.Input(input_shape)
    
    x = layers.Conv2D(64, (7, 7), activation="relu", strides=(2, 2))(x_input)
    x = layers.MaxPool2D((3, 3), strides=(2, 2))(x)
    
    x = conv_block(x, (3, 3), [64, 64, 256])
    x = identity_block(x, (3, 3), [64, 64, 256])
    x = identity_block(x, (3, 3), [64, 64, 256])
    
    x = conv_block(x, (3, 3), [128, 128, 512])
    x = identity_block(x, (3, 3), [128, 128, 512])
    x = identity_block(x, (3, 3), [128, 128, 512])
    x = identity_block(x, (3, 3), [128, 128, 512])
    
    x = conv_block(x, (3, 3), [256, 256, 1024])
    x = identity_block(x, (3, 3), [256, 256, 1024])
    x = identity_block(x, (3, 3), [256, 256, 1024])
    x = identity_block(x, (3, 3), [256, 256, 1024])
    x = identity_block(x, (3, 3), [256, 256, 1024])
    x = identity_block(x, (3, 3), [256, 256, 1024])
    
    x = conv_block(x, (3, 3), [512, 512, 2048])
    x = identity_block(x, (3, 3), [512, 512, 2048])
    x = identity_block(x, (3, 3), [512, 512, 2048])
    
    x = layers.AveragePooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(num_classes, activation="softmax")(x)
    
    return Model(inputs=x_input, outputs=x)
