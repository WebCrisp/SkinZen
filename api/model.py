import tensorflow as tf
import os

resnet101 = tf.keras.applications.ResNet101(include_top=False, weights="imagenet")

for layer in resnet101.layers:
  layer.trainable = False
  if layer.name == "conv3_block1_1_conv":
    break

IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3

def load_trained_model(weights_path):
    input = tf.keras.layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))

    # data augmentation
    a1 = tf.keras.layers.experimental.preprocessing.RandomRotation(0.4)(input)
    a2 = tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical")(a1)

    # resnet
    out = resnet101(a2)
    x = tf.keras.layers.GlobalMaxPooling2D()(out)

    # dense layer
    d1 = tf.keras.layers.Dense(256, activation="relu")(x)
    d2 = tf.keras.layers.Dense(128, activation="relu")(d1)
    d3 = tf.keras.layers.Dense(64, activation="relu")(d2)
    prediction = tf.keras.layers.Dense(3, activation="softmax")(d3)

    model = tf.keras.Model(inputs=input, outputs=prediction)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model.load_weights(weights_path)
    return model
