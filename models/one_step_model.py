import tensorflow as tf

class OneStep(tf.keras.Model):
    def __init__(self, model, vocabulary):
        super().__init__()
        self.model = model
        self.vocabulary = vocabulary
        self.ids_from_chars = tf.keras.layers.StringLookup(vocabulary=self.vocabulary, mask_token=None)
        self.chars_from_ids = tf.keras.layers.StringLookup(vocabulary=self.vocabulary, mask_token=None, invert=True)
        skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(values=[-float('inf')]*len(skip_ids), indices=skip_ids, dense_shape=[len(self.vocabulary)])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_one_step(self, inputs, temperature, states=None):
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()
        predicted_logits, states = self.model(inputs=input_ids, states=states, return_state=True)
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/temperature
        predicted_logits = predicted_logits + self.prediction_mask
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)
        predicted_chars = self.chars_from_ids(predicted_ids)
        return predicted_chars, states
    