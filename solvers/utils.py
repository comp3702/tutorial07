from typing import Dict


def dict_argmax(dictionary: Dict):
    max_value = max(dictionary.values()) # TODO handle multiple keys with the same max value
    for key, value in dictionary.items():
        if value == max_value:
            return key
