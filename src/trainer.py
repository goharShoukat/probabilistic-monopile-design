import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

from utils.load_config import load_config

config = load_config("config.yaml")
print(config)
