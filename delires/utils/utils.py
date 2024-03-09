import numpy as np
import json


def load_json(filename: str) -> dict:
    with open(filename, "r") as file:
        json_dict = json.load(file)
    return json_dict
           
def archive_kwargs(kwargs, path):
    json.dump(kwargs, open(path, "w"), indent="\t")