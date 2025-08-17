import tensorflow as tf
from tensorflow.keras import layers, Model

from config import config

def build_generator():
    inputs = layers.Input(shape=config.INPUT_SHAPE)

    # Initial conv
    x = layers.Conv2D(config.GEN_FILTERS, 9, padding='same')(inputs)
    x = layers.PReLU()(x)

    # Residual blocks
    for _ in range(config.GEN_RES_BLOCKS):
        x_res = x
        x = layers.Conv2D(config.GEN_FILTERS, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.PReLU()(x)
        x = layers.Conv2D(config.GEN_FILTERS, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x_res, x])

    # Upsampling
    x = layers.Conv2D(config.GEN_FILTERS * 4, 3, padding='same')(x)
    x = layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2))(x)
    x = layers.PReLU()(x)

    # Final output (float32)
    outputs = layers.Conv2D(3, 9, padding='same', activation='tanh')(x)

    return Model(inputs, outputs)

def build_discriminator():
    inputs = layers.Input(shape=config.HR_SHAPE)
    x = layers.Conv2D(64, 3, strides=1, padding='same')(inputs)
    x = layers.LeakyReLU(alpha=0.2)(x)
    for filters in [64, 128, 128, 256, 256, 512, 512]:
        x = layers.Conv2D(filters, 3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs)

def build_vgg_loss_model():
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=config.HR_SHAPE)
    feature_layer = vgg.get_layer('block5_conv4').output
    model = Model(inputs=vgg.input, outputs=feature_layer)
    model.trainable = False
    return model
