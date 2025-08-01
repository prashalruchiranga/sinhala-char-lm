import time
import tensorflow as tf
from utils import misc

project_root = misc.get_project_root()
configs = misc.load_config("config.yaml")
EXPORT_DIR_NAME = configs["export_dir_name"]
export_dir = project_root.joinpath(EXPORT_DIR_NAME)
reloaded_model = tf.saved_model.load(export_dir)

start = time.time()
states = None
next_char = tf.constant(['ජාතික'])
result = [next_char]

for n in range(1000):
    next_char, states = reloaded_model.generate_one_step(next_char, states=states, temperature=0.75)
    result.append(next_char)

result = tf.strings.join(result)
end = time.time()
print(result[0].numpy().decode('UTF-8'), '\n\n' + '_'*80)
print('\nRun time:', end - start)
