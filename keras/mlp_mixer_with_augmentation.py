import numpy as np
import tensorflow as tf
import keras
import time
import sys
import os
import tensorflow.keras.layers.experimental.preprocessing as preprocessing

# MLP Block
def MlpBlock(input_dim: int, hidden_dim: int):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_dim, activation='gelu'),
        tf.keras.layers.Dense(input_dim)
    ])

# Mixer Block
def MixerBlock(n_tokens: int, n_channels: int, token_mixer_hidden_dim: int, channel_mixer_hidden_dim, inputs):
    # inputs = tf.keras.Input(shape=(n_channels, n_tokens))
    y = tf.keras.layers.LayerNormalization()(inputs); 
    y = tf.keras.layers.Permute((2, 1))(y)
    # batch_size = y.shape[0]
    y = tf.reshape(y, (-1, n_tokens))
    y = MlpBlock(n_tokens, token_mixer_hidden_dim)(y)
    y = tf.reshape(y, (-1, n_channels, n_tokens))
    y = tf.keras.layers.Permute((2, 1))(y)
    x = tf.keras.layers.Add()([inputs, y])
    y = tf.keras.layers.LayerNormalization()(x); 
    y = tf.reshape(y, (-1, n_channels))
    y = MlpBlock(n_channels, channel_mixer_hidden_dim)(y)
    y = tf.reshape(y, (-1, n_tokens, n_channels))
    outputs = tf.keras.layers.Add()([x, y])
    return outputs
    # return tf.keras.Model(inputs, outputs)

def PatchAndProject(n_channels: int, patch_width: int):
    return tf.keras.layers.Conv2D(n_channels, patch_width, strides=patch_width)

def MlpMixer(patch_width: int, image_width: int, n_input_channels: int, n_channels: int, n_classes: int, n_mixer_blocks: int, token_mixer_hidden_dim: int, channel_mixer_hidden_dim: int, patch_embedding_hidden_dim: int):
    n_patches_side = int(image_width / patch_width)
    n_patches = int(n_patches_side ** 2)
    inputs = tf.keras.layers.Input([image_width,image_width,n_input_channels])
    
    # Augment and normalize
    # h = Augment()(inputs)
    h = tf.keras.layers.BatchNormalization(axis=-1)(inputs)

    # Normal mixer
    if patch_embedding_hidden_dim == 0:
        h = PatchAndProject(n_channels, patch_width)(h)
    else:
        h = PatchAndProject(patch_embedding_hidden_dim, patch_width)(h)
        h = tf.keras.layers.Activation(tf.keras.activations.relu)(h)
        h = PatchAndProject(n_channels, 1)(h)

    h = tf.reshape(h, (-1, n_patches, n_channels))
    for i in range(n_mixer_blocks):
        h = MixerBlock(n_patches, n_channels, token_mixer_hidden_dim, channel_mixer_hidden_dim, h)
    h = tf.keras.layers.LayerNormalization()(h)
    h = tf.reshape(h, (-1, n_patches_side, n_patches_side, n_channels))
    h = tf.keras.layers.GlobalAveragePooling2D(data_format="channels_last")(h)
    outputs = tf.keras.layers.Dense(n_classes)(h)
    ##########
    return tf.keras.Model(inputs, outputs)

def get_datagen_flow(data_path, n_examples, batch_size, image_width):
    data = tf.keras.preprocessing.image_dataset_from_directory(
        directory=data_path,
        labels="inferred",
        label_mode="int",
        class_names=None,
        color_mode="rgb",
        batch_size=n_examples,
        image_size=(image_width, image_width),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
        crop_to_aspect_ratio=False,
    )

    for batch in data:
        x_data, y_data = batch

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        # samplewise_center=True,
        # samplewise_std_normalization=True,
        rotation_range=90,
        width_shift_range=0.4,
        height_shift_range=0.4,
        brightness_range=(0.6, 1.2),
        shear_range=0.2,
        zoom_range=0.3,
        channel_shift_range=10.0,
        horizontal_flip=True,
        vertical_flip=True,
        )

    datagen.fit(x_data)
    
    return datagen.flow(x_data, y_data, batch_size=batch_size)

def get_dataset(data_path, batch_size, image_width):
    data = tf.keras.preprocessing.image_dataset_from_directory(
        directory=data_path,
        labels="inferred",
        label_mode="int",
        class_names=None,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(image_width, image_width),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
        crop_to_aspect_ratio=False,
    )

    return data

class Logger(object):
    def __init__(self, logfile_path):
        self.terminal = sys.stdout
        self.log = open(logfile_path, "a")
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass 

def run_experiment(
    train_data_path,
    n_train_examples,
    test_data_path,
    n_classes,
    save_path,
    logfile_path,
    n_mixer_blocks=2,
    token_mixer_hidden_dim=384,
    channel_mixer_hidden_dim=3072,
    n_input_channels=3,
    batch_size=64,
    image_width=128,
    patch_width=16,
    patch_embedding_dim=768,
    patch_embedding_hidden_dim=0,
    n_epochs=100,
    ):

    sys.stdout = Logger(logfile_path)

    if n_train_examples <= 100000:
        train_data = get_datagen_flow(train_data_path, n_train_examples, batch_size, image_width)
    else:
        train_data = get_dataset(train_data_path, batch_size, image_width)

    if test_data_path:
        test_data = get_dataset(test_data_path, batch_size, image_width)

    model = MlpMixer(
        patch_width, 
        image_width, 
        n_input_channels, 
        patch_embedding_dim, 
        n_classes, 
        n_mixer_blocks, 
        token_mixer_hidden_dim, 
        channel_mixer_hidden_dim, 
        patch_embedding_hidden_dim
        )

    # Create a callback that saves the model's weights
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    #                                                 save_weights_only=True,
    #                                                 verbose=1)

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )

    model.summary()

    for epoch in range(n_epochs):
        model.fit(train_data)
        print("Saving... Do not interrupt the program.")
        model.save(save_path)
        print(f"Saved to {save_path}")
        if test_data_path:
            _, acc = model.evaluate(test_data)
            print(f"Epoch {epoch + 1}: accuracy {acc}.")