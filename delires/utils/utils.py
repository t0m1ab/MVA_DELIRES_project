import os
import re 
from pathlib import Path
import json
from huggingface_hub import HfFileSystem, hf_hub_download

from delires.params import DIFFPIR_NETWOKRS, HF_REPO_ID, MODELS_PATH


def load_json(filename: str) -> dict:
    with open(filename, "r") as file:
        json_dict = json.load(file)
    return json_dict
           

def archive_kwargs(kwargs, path):
    json.dump(kwargs, open(path, "w"), indent="\t")
    
# Credit for this function goes to Jeff Atwood
def sorted_nicely(l): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def download_diffusion_model(path: str = None):
    """ 
    Download diffusion model listed in DIFFPIR_NETWORKS from HF_REPO_ID
    and stored them in path.
    """

    path = path if path is not None else MODELS_PATH
    Path(path).mkdir(parents=True, exist_ok=True)

    # list all .pt files in the HF repo
    fs = HfFileSystem()
    available_torch_networks = fs.glob(f"{HF_REPO_ID}/*.pt")

    # check if the networks are available
    filenames = []
    for network_name in DIFFPIR_NETWOKRS:
        filename = f"{HF_REPO_ID}/{network_name}.pt"
        if filename in available_torch_networks:
            filenames.append(f"{network_name}.pt")

    print(f"Downloading diffusion networks from {HF_REPO_ID}...")
    for network_filename in filenames:
        if os.path.isfile(os.path.join(path, network_filename)):
            print(f"File {network_filename} already exists in {path}")
        else:
            _ = hf_hub_download(
                repo_id=HF_REPO_ID,
                repo_type="model",
                filename=network_filename,
                local_dir=path,
            )


def main():
    download_diffusion_model()


if __name__ == "__main__":
    main()
