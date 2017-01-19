import functools
import json
from collections import OrderedDict

@functools.lru_cache(maxsize=8)
def load_config(path):
    with open(path) as config_file:
        return json.load(config_file, object_pairs_hook=OrderedDict)