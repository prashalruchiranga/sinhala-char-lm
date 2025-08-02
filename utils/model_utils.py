import numpy as np
import tensorflow as tf

def get_value_counts(text: str, decode=False) -> dict:
    chars = tf.strings.unicode_split(text, input_encoding='UTF-8')
    values, counts = np.unique(chars, return_counts=True)
    counts = map(lambda x: int(x), counts)
    if decode:
        values = map(lambda x: x.decode('UTF-8'), values)
    value_counts = dict(zip(values, counts))
    sorted_value_counts = dict(sorted(value_counts.items(), key=lambda x: x[1], reverse=True))
    return sorted_value_counts

def get_input_ids(text):
    vocab = sorted(set(text))
    chars = tf.strings.unicode_split(text, input_encoding='UTF-8')
    ids_from_chars = tf.keras.layers.StringLookup(vocabulary=vocab, mask_token=None)
    ids = ids_from_chars(chars)
    return ids, ids_from_chars.get_vocabulary()

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

def train_val_split(dataset, train_frac=0.8):
    dataset_size = tf.data.experimental.cardinality(dataset).numpy()
    train_size = int(train_frac * dataset_size)
    train_set = dataset.take(train_size)
    remaining = dataset.skip(train_size)
    return train_set, remaining

def save_model(model, export_dir):
    try:
        tf.saved_model.save(model, export_dir)
        message = f"Generator model saved to {export_dir}"
    except Exception as e:
        message = e
    return message
