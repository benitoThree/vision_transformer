import sys
sys.path.append('../keras')
from mlp_mixer_with_augmentation import run_experiment

run_experiment(
    train_data_path="/media/scratch/data/birds/train/",
    n_train_examples=100000,
    test_data_path="/media/scratch/data/birds/test/",
    n_classes=325,
    checkpoint_path="/home/benito/vision_transformer/experiments/mixer_keras_0_blocks_cp.ckpt",
    logfile_path="/home/benito/vision_transformer/experiments/mixer_keras_0_blocks.log",
    n_mixer_blocks=0,
    patch_embedding_dim=1536,
    patch_embedding_hidden_dim=768
    )