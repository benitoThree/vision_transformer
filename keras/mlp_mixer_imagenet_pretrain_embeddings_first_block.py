
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

test_data_path = sys.argv[3]
print("test_data_path:", test_data_path)

csv_save_file = sys.argv[4]
train_csv_save_file = csv_save_file + "_train.csv"
test_csv_save_file = csv_save_file + "_test.csv"
print(f"csv_save_files: {train_csv_save_file} and {test_csv_save_file}")

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

    # layer_norm = tf.keras.layers.LayerNormalization()
    # h = layer_norm(h)
    # if pretrained_params is not None:
    #     layer_norm_params = pretrained_params['pre_head_layer_norm']
    #     layer_norm.set_weights([layer_norm_params['scale'], layer_norm_params['bias']])
    #     layer_norm.trainable = False

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
n_mixer_blocks = 1
token_mixer_hidden_dim = 384
channel_mixer_hidden_dim = 3072

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory=train_data_path,
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

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory=test_data_path,
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(img_shape, img_shape),
    shuffle=False,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)

params = np.load(pretrained_path, allow_pickle=True)[()]
model = MlpMixer(filter_size, img_shape, n_channels, n_embedding_channels, n_mixer_blocks, token_mixer_hidden_dim, channel_mixer_hidden_dim, pretrained_params=params)

model.summary() # Sanity check

def embed(embedding_model, dataset, save_file):
    with open(save_file, "w") as out:
        for batch in dataset:
            examples, labels = batch
            x = (examples - 127.5) / 127.5
            batch_embeddings = embedding_model(x)
            for i, embedding in enumerate(batch_embeddings):
                print(embedding)
                out.write(f'{labels[i]},' + ','.join(embedding))

embed(model, train_data, train_csv_save_file)
embed(model, test_data, test_csv_save_file)


def sanity_check_csv_file(dataset, csv_file):
    failed = False
    line_count = 0
    for line in open(csv_file, "r"):
        line_count += 1
        line_split = line.split(',')
        line_label = int(line_split[0])
        line_embeddings = [float(elem) for elem in line_split[1:]]
        n_elems_in_line = len(line_split)
        if n_elems_in_line != 769:
            print(f"Something's wrong. Found {n_elems_in_line} elements (instead of 768) in line {line_count}.")
            failed = True
    
    expected_train_data_length = len(list(train_data))
    if line_count != expected_train_data_length:
        print(f"Something's wrong. Found {line_count} lines (instead of {expected_train_data_length}).")
        failed = True

    return failed

# Some sanity checks. Triggered if more arguments are passed than required. 
if len(sys.argv) > 5 and sys.argv[5] == "check":
    failed_test = False

    # Make sure no trainable parameter (params are only not trainable if pretrained parameters were loaded)
    trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    if (trainable_count > 0):
        print(f"Something's wrong. There are {trainable_count} trainable weights (instead of 0).")
        failed_test = True

    # Sanity check the csv files (correct number of entries per line, correct number of lines).
    failed_test = sanity_check_csv_file(train_data, train_csv_save_file)
    failed_test = sanity_check_csv_file(test_data, test_csv_save_file)
    
    if not failed_test:
        print("Sanity checks passed!")
