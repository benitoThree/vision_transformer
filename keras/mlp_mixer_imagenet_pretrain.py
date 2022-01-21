"""
All I need to do is to unravel the tree and inject it into my model.
In each of the model component constructors, I can pass a "params" object
so in MlpMixer, I would be passing in ckpt_dict[()]
then in in the for loop, at iteration i, I will call MixerBlock
To MixerBlock, I pass dict[f'MixerBlock_{i}']
To MlpBlock, I pass either dict['token_mixing']
All the above is recursive btw
And so on. That's really not terrible at all.
And if I pass None, then no setting.
And when I use pretrained, set to no training.
['MixerBlock_0': ['LayerNorm_0': ['bias', 'scale'], 'LayerNorm_1', 'channel_mixing': ['Dense_0': ['bias', 'kernel'], 'Dense_1'], 'token_mixing'], 'MixerBlock_1', 'MixerBlock_10', 'MixerBlock_11', 'MixerBlock_2', 'MixerBlock_3', 'MixerBlock_4', 'MixerBlock_5', 'MixerBlock_6', 'MixerBlock_7', 'MixerBlock_8', 'MixerBlock_9', 'head': ['bias', 'kernel'], 'pre_head_layer_norm': ['bias', 'scale'], 'stem': ['bias', 'kernel']]
stem is the convolution
"""

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

if len(sys.argv) < 3:
    print("Give pretrained path and checkpoint path!")



pretrained_path = sys.argv[1]
print("Getting pretrained parameters from", pretrained_path)
checkpoint_path = sys.argv[2]
print("Will save checkpoint to", checkpoint_path)
checkpoint_dir = os.path.dirname(checkpoint_path)

params = np.load(pretrained_path, allow_pickle=True)[()]

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
    directory="/media/scratch/data/birds/train/",
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

def normalize(image, label):
    image = tf.cast((image - 127.5)/127.5 ,tf.float32)
    return image, label

train_data = train_data.map(normalize)


for batch in train_data:
    x_train, y_train = batch

# datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#     # samplewise_center=True,
#     # samplewise_std_normalization=True,
#     # rotation_range=90,
#     # width_shift_range=0.4,
#     # height_shift_range=0.4,
#     brightness_range=(0.6, 1.2),
#     # shear_range=0.2,
#     zoom_range=0.3,
#     channel_shift_range=10.0,
#     horizontal_flip=True,
#     vertical_flip=True,
#     )

# datagen.fit(x_train)


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

test_data = test_data.map(normalize)

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

    ####### GET PATCH EMBEDDINGS #######
    
    h = tf.reshape(h, (-1, n_patches_side, n_patches_side, n_channels))
    h = tf.keras.layers.GlobalAveragePooling2D(data_format="channels_last")(h)
    outputs = tf.keras.layers.Dense(n_classes)(h)
    ##########
    return tf.keras.Model(inputs, outputs)

model = MlpMixer(filter_size, img_shape, n_channels, n_embedding_channels, n_classes, n_mixer_blocks, token_mixer_hidden_dim, channel_mixer_hidden_dim, pretrained_params=params)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

model.summary()



for epoch in range(100):
    model.fit(
        train_data,
        # datagen.flow(x_train, y_train, batch_size=batch_size), 
        callbacks=[cp_callback]
    )
    _, acc = model.evaluate(test_data)
    print(f"Epoch {epoch + 1}: accuracy {acc}.")