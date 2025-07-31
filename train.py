import tensorflow as tf
from pathlib import Path
from datasets import load_dataset
from utils import misc, model_utils

try:
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    strategy = tf.distribute.TPUStrategy(cluster_resolver)
    print("Using Tpu")
except:
    if tf.config.list_physical_devices('GPU'):
        strategy = tf.distribute.MirroredStrategy()
        print('Using GPU')
    else:
        strategy = tf.distribute.get_strategy()
        print("Using CPU")   

hf_dataset = "9wimu9/ada_derana_sinhala"

project_root = misc.get_project_root()
cache_dir = project_root.joinpath("dataset", "cache")
Path(cache_dir).mkdir(parents=True, exist_ok=True)
dataset = load_dataset(hf_dataset, cache_dir=cache_dir)


