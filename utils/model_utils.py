import re
import numpy as np
import tensorflow as tf
from tqdm import tqdm

def create_1d_list(n_d_list: list[list[str]]) -> list[str]:
    one_d_list = []
    for row in n_d_list:
        if len(row) > 1:
            for sub_row in row:
                one_d_list.append(sub_row)
        elif len(row) == 0:
            pass
        else:
            one_d_list.append(row[0])
    return one_d_list

def get_value_counts(text: str, decode=False) -> dict:
    chars = tf.strings.unicode_split(text, input_encoding='UTF-8')
    values, counts = np.unique(chars, return_counts=True)
    counts = map(lambda x: int(x), counts)
    if decode:
        values = map(lambda x: x.decode('UTF-8'), values)
    value_counts = dict(zip(values, counts))
    sorted_value_counts = dict(sorted(value_counts.items(), key=lambda x: x[1], reverse=True))
    return sorted_value_counts

def clean_text(text: str, replacement: str = "") -> str:
    print("Data cleaning in progress...")
    # lower english characters
    text = text.lower()
    # replace curly double quotes with straight double quotes
    text = re.sub(r"[\u201C\u201D]", "\u0022", text)
    # replace curly single quotes with straight single quotes
    text = re.sub(r"[\u2018\u2019]", "\u0027", text)
    # replace en-dash with dash
    text = re.sub(r"\u2013", "\u002D", text)
    # replace everything outside the allowed pattern with the replacement
    allowed_pattern = r"[a-z0-9\u0D80-\u0DFF\u200C\u200D!@#$%^&*()\[\]{}.,:;'\"<>?/\\|`~=_+ -]"
    # cleaned = ''.join(char if re.match(allowed_pattern, char) else replacement for char in text)
    cleaned_chars = []
    for char in tqdm(text, desc="Searching through text replacing OOV characters", unit="char"):
        if re.match(allowed_pattern, char):
            cleaned_chars.append(char)
        else:
            cleaned_chars.append(replacement)
    cleaned = ''.join(cleaned_chars)
    print("Data cleaning finished")
    return cleaned

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
