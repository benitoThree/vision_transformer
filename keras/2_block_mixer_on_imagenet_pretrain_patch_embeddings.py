
import numpy as np
import tensorflow as tf
import keras
import time
import sys
import os
import tensorflow.keras.layers.experimental.preprocessing as preprocessing
from thirdai import bolt


pretrained_path = "/home/benito/vision_transformer/pretrained_params/imagenet21k-Mixer-B_16.npy"
print("pretrained_path:", pretrained_path)

train_data_path = "/home/benito/datasets/backup/birds/train_100/"
print("train_data_path:", train_data_path)

test_data_path = "/home/benito/datasets/backup/birds/test_100/"
print("test_data_path:", test_data_path)

save_file = "/home/benito/vision_transformer/experiments/embeddings/pretrained_patches_100"
train_save_file = save_file + "_train"
test_save_file = save_file + "_test"
print(f"npy_save_files: {train_save_file} and {test_save_file}")

def PatchAndProject(n_channels: int, patch_width: int, pretrained_params, inputs):
    conv = tf.keras.layers.Conv2D(n_channels, patch_width, strides=patch_width) 
    outputs = conv(inputs)
    if pretrained_params is not None:
        conv.set_weights([pretrained_params['kernel'], pretrained_params['bias']])
        conv.trainable = False
    return outputs

def Model(
        patch_width: int, 
        image_width: int, 
        n_input_channels: int, 
        n_channels: int, 
        pretrained_params
        ):
    n_patches_side = int(image_width / patch_width)
    n_patches = int(n_patches_side ** 2)
    inputs = tf.keras.layers.Input([image_width,image_width,n_input_channels])
    
    # Normal mixer
    patch_and_project_params = pretrained_params['stem'] if pretrained_params is not None else None
    h = PatchAndProject(n_channels, patch_width, patch_and_project_params, inputs)
    
    outputs = tf.reshape(h, (-1, n_patches * n_channels))
    
    return tf.keras.Model(inputs, outputs)


filter_size = 16
n_channels = 3
n_embedding_channels = 768
feat_hash_dim = 100000 # for each patch separately
batch_size = 64 
img_shape = 128
n_patches = (img_shape**2)//(filter_size**2)
top_k = 5 # per patch non-zeros
n_mixer_blocks = 1
token_mixer_hidden_dim = 384
channel_mixer_hidden_dim = 3072

# train_data = tf.keras.preprocessing.image_dataset_from_directory(
#     directory=train_data_path,
#     labels="inferred",
#     label_mode="int",
#     class_names=None,
#     color_mode="rgb",
#     batch_size=batch_size,
#     image_size=(img_shape, img_shape),
#     shuffle=True,
#     seed=None,
#     validation_split=None,
#     subset=None,
#     interpolation="bilinear",
#     follow_links=False,
#     crop_to_aspect_ratio=False,
# )

train_data_for_datagen = tf.keras.preprocessing.image_dataset_from_directory(
    directory=train_data_path,
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

for batch in train_data_for_datagen:
    x_train, y_train = batch

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    brightness_range=(0.6, 1.2),
    # shear_range=0.2,
    zoom_range=0.3,
    channel_shift_range=10.0,
    horizontal_flip=True,
    vertical_flip=True,
    )

train_data = datagen.flow(x_train, y_train, batch_size=batch_size)

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
model = Model(filter_size, img_shape, n_channels, n_embedding_channels, pretrained_params=params)

model.summary() # Sanity check

def embed(embedding_model, dataset, save_file, batch_size, stop_after=None):
    image_labels = np.ndarray([0])
    image_patch_embeddings = np.ndarray([0, 64 * 768])
    n_examples = 0
    n_batches = 0
    for batch in dataset:
        if stop_after and n_examples >= stop_after:
            break
        examples, labels = batch
        x = (examples - 127.5) / 127.5
        batch_embeddings = embedding_model(x)
        image_patch_embeddings = np.append(image_patch_embeddings, batch_embeddings, axis=0)
        image_labels = np.append(image_labels, labels)
        n_examples += batch_size
        n_batches += 1
        print(f"Processed {n_examples} examples ({n_batches} batches).", end="\r")
    print(f"Finished processing {n_examples} examples ({n_batches} batches).", end="\n")
    np.save(save_file + "_embeddings", image_patch_embeddings)
    np.save(save_file + "_labels", image_labels)
    return image_patch_embeddings, image_labels

if os.path.exists(test_save_file + "_embeddings.npy" and test_save_file + "_labels.npy"):
    bolt_test_examples = np.load(test_save_file + "_embeddings.npy")
    bolt_test_labels = np.load(test_save_file + "_labels.npy")
    print("Loaded test embeddings")
else:
    bolt_test_examples, bolt_test_labels = embed(model, test_data, test_save_file, batch_size)
    print("Embedded test dataset")

if os.path.exists(train_save_file + "_embeddings.npy" and train_save_file + "_labels.npy"):
    bolt_train_examples = np.load(train_save_file + "_embeddings.npy")
    bolt_train_labels = np.load(train_save_file + "_labels.npy")
    print("Loaded train embeddings")
else:
    bolt_train_examples, bolt_train_labels = embed(model, train_data, train_save_file, batch_size, stop_after=57880)
    print("Embedded train dataset")


def MlpMixer(patch_width: int, image_width: int, n_input_channels: int, n_channels: int, n_classes: int, n_mixer_blocks: int, token_mixer_hidden_dim: int, channel_mixer_hidden_dim: int):
    n_patches_side = int(image_width // patch_width)
    n_patches = int(n_patches_side ** 2)
    inputs = tf.keras.layers.Input([n_patches * n_channels])
    h = tf.reshape(h, (-1, n_patches, n_channels))
    for i in range(n_mixer_blocks):
        h = MixerBlock(n_patches, n_channels, token_mixer_hidden_dim, channel_mixer_hidden_dim, h)
    h = tf.keras.layers.LayerNormalization()(h)
    h = tf.reshape(h, (-1, n_patches_side, n_patches_side, n_channels))
    h = tf.keras.layers.GlobalAveragePooling2D(data_format="channels_last")(h)
    outputs = tf.keras.layers.Dense(n_classes)(h)
    ##########
    return tf.keras.Model(inputs, outputs)

model = MlpMixer(
        patch_width=16, 
        image_width=128, 
        n_input_channels=3, 
        n_channels=768, 
        n_classes=100, 
        n_mixer_blocks=2, 
        token_mixer_hidden_dim=384, 
        channel_mixer_hidden_dim=3072, 
        )

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

mixer_train_data = tf.data.Dataset.from_tensor_slices((bolt_train_examples, bolt_train_labels)).batch(batch_size)
mixer_test_data = tf.data.Dataset.from_tensor_slices((bolt_test_examples, bolt_test_examples)).batch(batch_size)


for i in range(10):
    model.fit(mixer_train_data)
    model.evaluate(mixer_test_data)
        
