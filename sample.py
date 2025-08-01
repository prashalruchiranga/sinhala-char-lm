import tensorflow as tf
from utils import misc

def sample(initial_input_str: str, sample_size: int, temperature: float = 1.0):
    configs = misc.load_config("config.yaml")
    EXPORT_DIR_NAME = configs["export_dir_name"]
    project_root = misc.get_project_root()
    saved_model_path = project_root.joinpath(EXPORT_DIR_NAME)
    reloaded_model = tf.saved_model.load(saved_model_path)
    states = None
    next_char = tf.constant([initial_input_str])
    result = [next_char]
    for n in range(sample_size):
        next_char, states = reloaded_model.generate_one_step(inputs=next_char, temperature=temperature, states=states)
        result.append(next_char)
    result = tf.strings.join(result)
    return result

if __name__ == "__main__":
    generated_sample = sample(initial_input_str='k', sample_size=100, temperature=0.8)
    print(generated_sample[0].numpy().decode('UTF-8'))
