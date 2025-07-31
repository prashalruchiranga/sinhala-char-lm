import re
import numpy as np
import tensorflow as tf

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
    cleaned = ''.join(char if re.match(allowed_pattern, char) else replacement for char in text)
    return cleaned

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

