import tensorflow as tf
from utils import misc
from types import SimpleNamespace
import argparse

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
    # Load configuration parameters
    config_dict = misc.load_config("config.yaml")
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Provide initial input string, sample size and temperature")
    parser.add_argument("--string", type=str, required=True, help="Initial input string")
    parser.add_argument("--length", type=int, required=True, help="Sample size")
    parser.add_argument("--temperature", type=float, required=True, help="Temperature")
    args = parser.parse_args()
    # Sample text
    generated_sample = sample(configs=config_dict, initial_input_str=args.string, sample_size=args.length, temperature=args.temperature)
    print(generated_sample[0].numpy().decode('UTF-8'))
