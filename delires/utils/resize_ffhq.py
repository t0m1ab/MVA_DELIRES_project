import os
from pathlib import Path
from tqdm import tqdm
import argparse
from PIL import Image

from delires.utils import utils
from delires.params import CLEAN_DATA_PATH


def resize_ffhq(path: str, folder_idx: int, filename: str, size: int):
    """ Download FFHQ images. """

    path = path if path is not None else os.path.join(str(Path(CLEAN_DATA_PATH).parent), "ffhq/")

    resized_path = os.path.join(path, f"{folder_idx}000_{size}x{size}/")
    ffhq_path = os.path.join(path, f"{folder_idx}000/")

    if not os.path.exists(ffhq_path):
        raise ValueError(f"Invalid path: {ffhq_path}. Must be a valid directory.")
    
    if filename is None:
        files = utils.listdir(ffhq_path)
        if len(files) == 0:
            raise ValueError(f"No files found in path: {ffhq_path}.")
    else:
        if not os.path.isfile(os.path.join(ffhq_path, filename)):
            raise ValueError(f"Invalid filename: {filename}. Must be a valid file in path: {ffhq_path}.")
        files = [filename]

    Path(resized_path).mkdir(parents=True, exist_ok=True)
    
    pbar = tqdm(files, desc=f"Resizing {len(files)} images", total=len(files))
    for f in pbar:
        img = Image.open(os.path.join(ffhq_path, f))
        img = img.resize((size, size))
        img.save(os.path.join(resized_path, f))


def main():

    parser = argparse.ArgumentParser(description="Resize FFHQ images")
    parser.add_argument(
        "-p",
        "--path", 
        type=str,
        default=None,
        help="path to the folder containing the images to resize."
    )
    parser.add_argument(
        "-f",
        "--folder",
        type=int, 
        help="idx of the folder. For ex: idx=61 for folder 61000/."
    )
    parser.add_argument(
        "-n",
        "--name", 
        type=str, 
        default=None, 
        help="name of the image in path to resize, if None, resize all images in path."
    )
    parser.add_argument(
        "-s",
        "--size", 
        type=int, 
        default=256, 
        help="target size for resizing the images",
    )
    args = parser.parse_args()

    resize_ffhq(
        path=args.path,
        folder_idx=args.folder,
        filename=args.name,
        size=args.size,
    )


if __name__ == "__main__":
    main()
    # RESIZE ALL IMAGES IN FOLDER 60000/ TO 256x256: python resize_ffhq.py -f 60
    # RESIZE ONLY IMAGE 60000/60001.png TO 256x256: python resize_ffhq.py -f 60 -n 60001.png -s 256