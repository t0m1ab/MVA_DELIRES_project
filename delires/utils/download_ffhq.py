import os
from pathlib import Path
import argparse
import gdown

from delires.params import CLEAN_DATA_PATH

TRAIN_FOLDER_INDEXES = [k for k in range(60)]
VAL_FOLDER_INDEXES = [k for k in range(60,70)]

FOLDER_INDEX_TO_URL = {
    60: "https://drive.google.com/drive/folders/1ni02mkE3q8gkfoDJIWjkQ0SZMbYPwY5l",
    61: "https://drive.google.com/drive/folders/19rwTm2K6iCA9KEqBqqtkf0GktDt6i_vn",
    62: "https://drive.google.com/drive/folders/1zYYycp35OCApbN0ZaVEfJWsFfV69kLcb",
    69: "https://drive.google.com/drive/folders/1zF65a7mSulFh1rUKpd1N5cPkOWZvJm86",
}


def download_ffhq(split: str, folder_idx: int, path: str = None):
    """ Download FFHQ images. """

    if not folder_idx in FOLDER_INDEX_TO_URL:
        raise ValueError(f"Invalid folder index: {folder_idx}. Must be one of {VAL_FOLDER_INDEXES}")

    path = path if path is not None else os.path.join(str(Path(CLEAN_DATA_PATH).parent), "ffhq/")
    Path(path).mkdir(parents=True, exist_ok=True)

    gdown.download_folder(
        url=FOLDER_INDEX_TO_URL[folder_idx], 
        output=os.path.join(path, f"ffhq_{folder_idx}000.zip"), 
        use_cookies=False,
        remaining_ok=True, # for large downloads
    )


def main():

    SPLITS = ["train", "val"]

    parser = argparse.ArgumentParser(description="Download FFHQ dataset")
    parser.add_argument(
        "-s",
        "--split", 
        type=str, 
        default="val", 
        help=f"Split to download. Must be one of {SPLITS}",
    )
    parser.add_argument(
        "-f",
        "--folder", 
        type=int, 
        default=None, 
        help="Folder index to download. If None, download the entire split. For ex: if idx=62 then download folder 62000/."
    )
    args = parser.parse_args()
    
    if args.split == "train":
        raise ValueError("The training split is too large to be downloaded. Please use the official FFHQ dataset.")
    elif args.split == "val":
        if args.folder is None:
            indexes = VAL_FOLDER_INDEXES
        else:
            indexes = [args.folder]
        for folder_idx in indexes:
            download_ffhq(split=args.split, folder_idx=folder_idx)
    else:
        raise ValueError(f"Invalid split: {args.split}. Must be one of {SPLITS}")


if __name__ == "__main__":
    main()
    # python download_ffhq.py -s val -f 61