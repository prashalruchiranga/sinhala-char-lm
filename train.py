import os
import tensorflow as tf
from datasets import load_dataset
from utils import misc, model_utils
from models.gru import GRUModel
from models.one_step_model import OneStep

def train_char_lm(configs: dict):
    HF_DATASET = configs["huggingface_dataset"]
    SEQ_LENGTH = configs["seq_length"]
    BUFFER_SIZE = configs["buffer_size"]
    SEED = configs["seed"]
    BATCH_SIZE = configs["batch_size"]
    EMBEDDING_DIM = configs["embedding_dim"]
    RNN_UNITS = configs["rnn_units"]
    EPOCHS = configs["epochs"]
    LEARNING_RATE = configs["learning_rate"]
    TRAIN_FRAC = configs["train_frac"]
    EXPORT_DIR_NAME = configs["export_dir_name"]
    DATASET_FRAC = configs["dataset_frac"]

    strategy = misc.get_distribute_strategy()

    project_root = misc.get_project_root()
    cache_dir = project_root.joinpath("dataset", "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(HF_DATASET, cache_dir=cache_dir)

    raw_ds = dataset["train"]
    content = raw_ds["content"]
    text_list = model_utils.create_1d_list(content)
    text = " ".join(text_list)

    if 0 < DATASET_FRAC < 1.0:
        limit = int(DATASET_FRAC * len(text))
        text = text[:limit]
    elif DATASET_FRAC <= 0 or DATASET_FRAC > 1.0:
        raise ValueError("dataset_frac must be a value between 0 and 1 (0, 1], includes 1.0")
    
    cleaned_text = model_utils.clean_text(text)

    processed_dir = project_root.joinpath("dataset", "processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_file = processed_dir.joinpath("cleaned.txt")
    with open(processed_file, 'w') as file:
        file.write(cleaned_text)
    print(f"Saved cleaned text to {processed_file}")

    print("Tokenizing text in progress...")
    input_ids, vocabulary = model_utils.get_input_ids(cleaned_text)

    vocab_size = len(vocabulary)
    print("Tokenization finished")

    ids_dataset = tf.data.Dataset.from_tensor_slices(input_ids)
    sequences = ids_dataset.batch(SEQ_LENGTH+1, drop_remainder=True)
    full_dataset = sequences.map(model_utils.split_input_target)
    shuffled_dataset = full_dataset.shuffle(buffer_size=BUFFER_SIZE, reshuffle_each_iteration=False, seed=SEED)
    train_set, val_set = model_utils.train_val_split(dataset=shuffled_dataset, train_frac=TRAIN_FRAC)
    train_set = train_set.batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    val_set = val_set.batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

    checkpoints_dir = project_root.joinpath("training_checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_prefix = os.path.join(str(checkpoints_dir), "ckpt_{epoch}.weights.h5")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=2,
        verbose=1,
        restore_best_weights=True
    )

    with strategy.scope():
        model = GRUModel(vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM, rnn_units=RNN_UNITS)
        adam = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer=adam, loss=loss)

    model.fit(train_set, validation_data=val_set, epochs=EPOCHS, 
              callbacks=[checkpoint_callback, early_stopping])

    ids_from_chars = tf.keras.layers.StringLookup(vocabulary=vocabulary, mask_token=None)
    chars_from_ids = tf.keras.layers.StringLookup(vocabulary=vocabulary, mask_token=None, invert=True)
    one_step_model = OneStep(model=model, chars_from_ids=chars_from_ids, ids_from_chars=ids_from_chars)

    # Call one step model at least twice(n=2) with actual input data so that generate_one_step function will be 
    # traced (unless the function will not be saved)
    states = None
    temp = tf.constant(0.75)
    next_char = tf.constant(['a'])
    for n in range(100):
        next_char, states = one_step_model.generate_one_step(inputs=next_char, temperature=temp, states=states)

    export_dir = project_root.joinpath(EXPORT_DIR_NAME)
    try:
        tf.saved_model.save(one_step_model, export_dir)
        print(f"Generator model saved to {export_dir}")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    config_dict = misc.load_config("config.yaml")
    train_char_lm(config_dict)
