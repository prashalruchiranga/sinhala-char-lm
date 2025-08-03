from pathlib import Path
import tensorflow as tf
import yaml

def get_project_root() -> Path:
    return Path(__file__).parent.parent

def get_distribute_strategy(tpu="local"):
    try:
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu)
        tf.config.experimental_connect_to_cluster(cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        strategy = tf.distribute.TPUStrategy(cluster_resolver)
        print("Using TPU for training")
    except:
        if tf.config.list_physical_devices('GPU'):
            strategy = tf.distribute.MirroredStrategy()
            print("Using GPU for training")
        else:
            strategy = tf.distribute.get_strategy()
            print("Using CPU for training")
    return strategy

def load_config(path: str) -> dict:
    with open(path) as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise
