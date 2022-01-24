import tensorflow as tf
import sys
import os

if len(sys.argv) < 2:
    print("Give a checkpoint path!")

checkpoint_path = sys.argv[1]
print("Will save checkpoint to", checkpoint_path)
checkpoint_dir = os.path.dirname(checkpoint_path)

def WideMixer(image_width, patch_width, n_channels, patch_project_dim, n_classes):
    inputs = tf.keras.layers.Input([image_width, image_width, n_channels])
    h = (inputs - 127.5) / 127.5
    # Patch and project
    h = tf.keras.layers.Conv2D(patch_project_dim, patch_width, strides=patch_width)(h)
    print(h.shape)
    # Fully connected 1
    n_patches = int(image_width / patch_width) ** 2
    fully_connected_dim = n_patches * patch_project_dim
    h = tf.reshape(h, (-1, fully_connected_dim))
    print(h.shape)
    h = tf.keras.layers.Dense(fully_connected_dim, activation='gelu')(h)
    print(h.shape)
    # # Fully connected 2
    h = tf.keras.layers.Dense(fully_connected_dim, activation='gelu')(h)
    #h = tf.keras.layers.Dense(fully_connected_dim / 4, activation='gelu')(h)
    #h = tf.keras.layers.Dense(fully_connected_dim, activation='gelu')(h)
    # print(h.shape)
    # Pooling
    h = tf.reshape(h, [-1, n_patches, patch_project_dim])
    print(h.shape)
    h = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_last")(h)
    print(h.shape)
    # Classifying
    outputs = tf.keras.layers.Dense(n_classes)(h)
    print(h.shape)
    return tf.keras.Model(inputs, outputs)

filter_size = 16
n_channels = 3
n_embedding_channels = 768
feat_hash_dim = 100000 # for each patch separately
batch_size = 64 # also tried 128
img_shape = 128
n_patches = (128*128)//(16*16)
top_k = 5 # per patch non-zeros
n_classes = 325

train_data_for_datagen = tf.keras.preprocessing.image_dataset_from_directory(
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

for batch in train_data_for_datagen:
    x_train_for_datagen, y_train_for_datagen = batch

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    # samplewise_center=True,
    # samplewise_std_normalization=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    channel_shift_range=5.0,
    horizontal_flip=True,
    )

# datagen.fit(x_train_for_datagen)

def normalize(image, label):
    image = tf.cast((image - 127.5)/127.5 ,tf.float32)
    return image, label

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory="/media/scratch/data/birds/train/",
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

train_data = train_data.map(normalize)

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

test_data_for_datagen = test_data
test_data = test_data.map(normalize)



model = WideMixer(img_shape, filter_size, n_channels, n_embedding_channels, n_classes)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

for epoch in range(100):
    model.fit(
        # train_data
        datagen.flow(x_train_for_datagen, y_train_for_datagen, batch_size=batch_size), callbacks=[cp_callback]
        )
    # for batch in train_data:
    #     imgs = batch[0]
    #     labels = batch[1]
    #     model.train_on_batch(imgs, labels)
    ##
    _, acc = model.evaluate(test_data_for_datagen)
    print(f"Epoch {epoch + 1}: accuracy {acc}.")
