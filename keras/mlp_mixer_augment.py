import numpy as np
import tensorflow as tf
import keras
import time
import sys
from tensorflow.python.ops.numpy_ops import np_config
import os
import tensorflow.keras.layers.experimental.preprocessing as preprocessing
print(tf.version.VERSION)
np_config.enable_numpy_behavior()

if len(sys.argv) < 2:
    print("Give a checkpoint path!")

checkpoint_path = sys.argv[1]
print("Will save checkpoint to", checkpoint_path)
checkpoint_dir = os.path.dirname(checkpoint_path)

filter_size = 16
n_channels = 3
n_embedding_channels = 768
feat_hash_dim = 100000 # for each patch separately
batch_size = 64
img_shape = 128
n_patches = (128*128)//(16*16)
top_k = 5 # per patch non-zeros
n_classes = 325
n_mixer_blocks = 4
token_mixer_hidden_dim = 384
channel_mixer_hidden_dim = 3072

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory="/media/scratch/data/birds/train/",
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=100000,
    image_size=(img_shape, img_shape),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)

for batch in train_data:
    x_train, y_train = batch

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

datagen.fit(x_train)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory="/media/scratch/data/birds/test/",
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(img_shape, img_shape),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)

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

def MlpMixer(patch_width: int, image_width: int, n_input_channels: int, n_channels: int, n_classes: int, n_mixer_blocks: int, token_mixer_hidden_dim: int, channel_mixer_hidden_dim: int):
    n_patches_side = int(image_width / patch_width)
    n_patches = int(n_patches_side ** 2)
    inputs = tf.keras.layers.Input([image_width,image_width,n_input_channels])
    
    # Augment and normalize
    # h = Augment()(inputs)
    h = tf.keras.layers.BatchNormalization(axis=-1)(inputs)

    # Normal mixer
    h = PatchAndProject(n_channels, patch_width)(h)
    h = tf.reshape(h, (-1, n_patches, n_channels))
    for i in range(n_mixer_blocks):
        h = MixerBlock(n_patches, n_channels, token_mixer_hidden_dim, channel_mixer_hidden_dim, h)
    h = tf.keras.layers.LayerNormalization()(h)
    h = tf.reshape(h, (-1, n_patches_side, n_patches_side, n_channels))
    h = tf.keras.layers.GlobalAveragePooling2D(data_format="channels_last")(h)
    outputs = tf.keras.layers.Dense(n_classes)(h)
    ##########
    return tf.keras.Model(inputs, outputs)

model = MlpMixer(filter_size, img_shape, n_channels, n_embedding_channels, n_classes, n_mixer_blocks, token_mixer_hidden_dim, channel_mixer_hidden_dim)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )



for epoch in range(100):
    model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), callbacks=[cp_callback])
    # for batch in train_data:
    #     imgs = batch[0]
    #     labels = batch[1]
    #     model.train_on_batch(imgs, labels)
    ##
    _, acc = model.evaluate(test_data)
    print(f"Epoch {epoch + 1}: accuracy {acc}.")