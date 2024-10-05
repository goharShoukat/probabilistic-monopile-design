import yaml
import os


PATH = "src/config/"


def load_config(config_name: str):
    with open(os.path.join(PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config
