
import numpy as np
import tensorflow as tf
import keras
import time
import sys
import os
import tensorflow.keras.layers.experimental.preprocessing as preprocessing

pretrained_path = sys.argv[1]
print("pretrained_path:", pretrained_path)

train_data_path = sys.argv[2]
print("train_data_path:", train_data_path)

npy_save_file = sys.argv[3]
print("npy_save_file:", npy_save_file)

# MLP Block
def MlpBlock(input_dim: int, hidden_dim: int, pretrained_params, inputs):
    dense_0 = tf.keras.layers.Dense(hidden_dim, activation='gelu')
    x = dense_0(inputs)
    dense_1 = tf.keras.layers.Dense(input_dim)
    outputs = dense_1(x)
    if pretrained_params is not None:
        dense_params_0 = pretrained_params['Dense_0']
        dense_0.set_weights([dense_params_0['kernel'], dense_params_0['bias']])
        dense_0.trainable = False
        dense_params_1 = pretrained_params['Dense_1']
        dense_1.set_weights([dense_params_1['kernel'], dense_params_1['bias']])
        dense_1.trainable = False
    return outputs

# Mixer Block
def MixerBlock(n_tokens: int, n_channels: int, token_mixer_hidden_dim: int, channel_mixer_hidden_dim, pretrained_params, inputs):
    # inputs = tf.keras.Input(shape=(n_channels, n_tokens))
    layer_norm_0 = tf.keras.layers.LayerNormalization()
    y = layer_norm_0(inputs)
    if pretrained_params is not None:
        layer_params_0 = pretrained_params['LayerNorm_0']
        layer_norm_0.set_weights([layer_params_0['scale'], layer_params_0['bias']])
        layer_norm_0.trainable = False

    y = tf.keras.layers.Permute((2, 1))(y)
    # batch_size = y.shape[0]
    y = tf.reshape(y, (-1, n_tokens))
    
    token_mixer_params = pretrained_params['token_mixing'] if pretrained_params is not None else None
    y = MlpBlock(n_tokens, token_mixer_hidden_dim, token_mixer_params, y)
    y = tf.reshape(y, (-1, n_channels, n_tokens))
    y = tf.keras.layers.Permute((2, 1))(y)
    x = tf.keras.layers.Add()([inputs, y])

    layer_norm_1 = tf.keras.layers.LayerNormalization()
    y = layer_norm_1(x); 
    if pretrained_params is not None:
        layer_params_1 = pretrained_params['LayerNorm_1']
        layer_norm_1.set_weights([layer_params_1['scale'], layer_params_1['bias']])
        layer_norm_1.trainable = False
    
    y = tf.reshape(y, (-1, n_channels))

    channel_mixer_params = pretrained_params['channel_mixing'] if pretrained_params is not None else None
    y = MlpBlock(n_channels, channel_mixer_hidden_dim, channel_mixer_params, y)
    y = tf.reshape(y, (-1, n_tokens, n_channels))
    outputs = tf.keras.layers.Add()([x, y])
    return outputs
    # return tf.keras.Model(inputs, outputs)

def PatchAndProject(n_channels: int, patch_width: int, pretrained_params, inputs):
    conv = tf.keras.layers.Conv2D(n_channels, patch_width, strides=patch_width) 
    outputs = conv(inputs)
    if pretrained_params is not None:
        conv.set_weights([pretrained_params['kernel'], pretrained_params['bias']])
        conv.trainable = False
    return outputs

def MlpMixer(
        patch_width: int, 
        image_width: int, 
        n_input_channels: int, 
        n_channels: int, 
        n_classes: int, 
        n_mixer_blocks: int, 
        token_mixer_hidden_dim: int, 
        channel_mixer_hidden_dim: int,
        pretrained_params
        ):
    n_patches_side = int(image_width / patch_width)
    n_patches = int(n_patches_side ** 2)
    inputs = tf.keras.layers.Input([image_width,image_width,n_input_channels])
    
    # Normal mixer
    patch_and_project_params = pretrained_params['stem'] if pretrained_params is not None else None
    h = PatchAndProject(n_channels, patch_width, patch_and_project_params, inputs)
    
    h = tf.reshape(h, (-1, n_patches, n_channels))
    for i in range(n_mixer_blocks):
        mixer_block_params = pretrained_params[f'MixerBlock_{i}'] if pretrained_params is not None else None
        h = MixerBlock(n_patches, n_channels, token_mixer_hidden_dim, channel_mixer_hidden_dim, mixer_block_params, h)

    layer_norm = tf.keras.layers.LayerNormalization()
    h = layer_norm(h)
    if pretrained_params is not None:
        layer_norm_params = pretrained_params['pre_head_layer_norm']
        layer_norm.set_weights([layer_norm_params['scale'], layer_norm_params['bias']])
        layer_norm.trainable = False

    outputs = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_last")(h)
    return tf.keras.Model(inputs, outputs)


filter_size = 16
n_channels = 3
n_embedding_channels = 768
feat_hash_dim = 100000 # for each patch separately
batch_size = 64 
img_shape = 224
n_patches = (img_shape**2)//(filter_size**2)
top_k = 5 # per patch non-zeros
n_classes = 325
n_mixer_blocks = 12
token_mixer_hidden_dim = 384
channel_mixer_hidden_dim = 3072

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory=train_data_path,
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=batch_size, #TODO: CHANGE TO 100K if not datagen!!!
    image_size=(img_shape, img_shape),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)

train_data_for_datagen = tf.keras.preprocessing.image_dataset_from_directory(
    directory=train_data_path,
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=100000, #TODO: CHANGE TO 100K if not datagen!!!
    image_size=(img_shape, img_shape),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)

for batch in train_data_for_datagen:
    x_train, y_train = batch

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    # samplewise_center=True,
    # samplewise_std_normalization=True,
    # rotation_range=90,
    # width_shift_range=0.4,
    # height_shift_range=0.4,
    brightness_range=(0.6, 1.2),
    # shear_range=0.2,
    zoom_range=0.3,
    channel_shift_range=10.0,
    horizontal_flip=True,
    vertical_flip=True,
    )

params = np.load(pretrained_path, allow_pickle=True)[()]
model = MlpMixer(filter_size, img_shape, n_channels, n_embedding_channels, n_classes, n_mixer_blocks, token_mixer_hidden_dim, channel_mixer_hidden_dim, pretrained_params=params)

outputs = np.ndarray([0, n_embedding_channels])
for batch in train_data:
    x = (batch[0] - 127.5) / 127.5
    new_output = model(x)
    np.append(outputs, new_output, 0)
    print(outputs.shape)

for batch in datagen.flow(x_train, y_train, batch_size=batch_size):
    x = (batch[0] - 127.5) / 127.5
    new_output = model(x)
    np.append(outputs, new_output, 0)
    print(outputs.shape)

np.save(npy_save_file, outputs)