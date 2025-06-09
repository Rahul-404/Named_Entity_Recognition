import os
from typing import Dict

import yaml


def read_config(file_name: str) -> Dict:
    config_path = os.path.join(os.getcwd(), file_name)
    with open(config_path) as config:
        content = yaml.safe_load(config)

    return content