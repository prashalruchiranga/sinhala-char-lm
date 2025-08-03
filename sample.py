import tensorflow as tf
from utils import misc
from types import SimpleNamespace

def sample(configs: dict, initial_input_str: str, sample_size: int, temperature: float = 1.0):
    # Load configuration parameters
    configs = SimpleNamespace(**configs)

    # Realod the saved model
    project_root = misc.get_project_root()
    saved_model_path = project_root.joinpath(configs.export_dir_name)
    reloaded_model = tf.saved_model.load(saved_model_path)

    # Generate one character per iteration for 'sample_size' iterations
    states = None
    next_char = tf.constant([initial_input_str])
    result = [next_char]
    for n in range(sample_size):
        next_char, states = reloaded_model.generate_one_step(inputs=next_char, temperature=temperature, states=states)
        result.append(next_char)
    result = tf.strings.join(result)
    return result

if __name__ == "__main__":
    config_dict = misc.load_config("config.yaml")
    generated_sample = sample(configs=config_dict, initial_input_str='k', sample_size=100, temperature=0.8)
    print(generated_sample[0].numpy().decode('UTF-8'))
