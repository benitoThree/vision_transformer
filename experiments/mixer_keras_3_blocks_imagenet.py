import sys
sys.path.append('../keras')
from mlp_mixer_with_augmentation import run_experiment

run_experiment(
    train_data_path="/home/benito/datasets/imagenet/ILSVRC/Data/CLS-LOC/train/",
    n_train_examples=1500000,
    test_data_path="",
    n_classes=1000,
    save_path="/home/benito/vision_transformer/experiments/mixer_keras_3_blocks_imagenet",
    logfile_path="/home/benito/vision_transformer/experiments/mixer_keras_3_blocks_imagenet.log",
    n_mixer_blocks=3,
    patch_embedding_dim=768,
    n_epochs=100,
    )
