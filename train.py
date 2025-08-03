import os
import tensorflow as tf
from datasets import load_dataset
from utils import misc, model_utils
from models.gru import GRUModel
from models.one_step_model import OneStep
from utils.cleaning import Cleaner
from types import SimpleNamespace
from pathlib import Path
import sys

def train_char_lm(configs: dict, text_file_path: Path):
    # Load configuration parameters
    configs = SimpleNamespace(**configs)
    es_configs = SimpleNamespace(**configs.early_stopping)

    # Load cleaned text data to memory
    with open(text_file_path, "r") as file:
        text_data = file.read()

    # Convert the strings to numerical representation
    print("Tokenizing text in progress...")
    input_ids, vocabulary = model_utils.get_input_ids(text_data)
    vocab_size = len(vocabulary)
    print("Tokenization finished")

    # Split the vectorized text into train and validation sets
    ids_dataset = tf.data.Dataset.from_tensor_slices(input_ids)
    seq_length = configs.seq_length + 1
    sequences = ids_dataset.batch(seq_length, drop_remainder=True)
    full_dataset = sequences.map(model_utils.split_input_target)
    shuffled_dataset = full_dataset.shuffle(buffer_size=configs.buffer_size, reshuffle_each_iteration=False, seed=configs.seed)
    train_set, val_set = model_utils.train_val_split(dataset=shuffled_dataset, train_frac=configs.train_frac)
    train_set = train_set.batch(configs.batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    val_set = val_set.batch(configs.batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

    # Define checkpoint and early stopping callbacks
    project_root = misc.get_project_root()
    checkpoints_dir = project_root.joinpath(configs.checkpoints_dir_name)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_prefix = os.path.join(str(checkpoints_dir), "ckpt_{epoch}.weights.h5")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=es_configs.monitor,
        min_delta=es_configs.min_delta,
        patience=es_configs.patience,
        verbose=es_configs.verbose,
        restore_best_weights=es_configs.restore_best_weights
    )

    # Set distribute strategy for training
    strategy = misc.get_distribute_strategy()

    # Create and compile the model within the distribution strategy scope
    with strategy.scope():
        model = GRUModel(vocab_size=vocab_size, embedding_dim=configs.embedding_dim, rnn_units=configs.rnn_units)
        adam = tf.keras.optimizers.Adam(learning_rate=configs.learning_rate)
        loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer=adam, loss=loss)

    # Train the model
    history = model.fit(train_set, validation_data=val_set, epochs=configs.epochs, callbacks=[checkpoint_callback, early_stopping])
    
    # Define a model that makes a single step prediction
    one_step_model = OneStep(model=model, vocabulary=vocabulary)

    # Call one step model at least twice(n=2) with dummy data so that generate_one_step function will be traced
    states = None
    temp = tf.constant(0.75)
    next_char = tf.constant(['a'])
    for n in range(100):
        next_char, states = one_step_model.generate_one_step(inputs=next_char, temperature=temp, states=states)

    # Export the generator
    export_dir = project_root.joinpath(configs.export_dir_name)
    message = model_utils.save_model(one_step_model, export_dir)
    print(message)
    return history

def clean_dataset(configs: dict) -> Path:
    # Load configuration parameters
    configs = SimpleNamespace(**configs)
    # Clean the data and save to a local file
    project_root = misc.get_project_root()
    data_dir = project_root.joinpath(configs.dataset_dir_name)
    cleaner = Cleaner(data_export_dir=data_dir)
    cleaned_file_path = cleaner(hf_dataset=configs.huggingface_dataset, dataset_frac=configs.dataset_frac)
    return cleaned_file_path

if __name__ == "__main__":
    config_dict = misc.load_config("config.yaml")
    # Provide a cleaned text file unless you want to run the cleaning process from scratch
    # using the Hugging Face dataset defined in the config.yaml file
    try:
        cleaned_file_path = sys.argv[1]
    except IndexError:
        cleaned_file_path = clean_dataset(config_dict)
    history = train_char_lm(config_dict, cleaned_file_path)
    
    