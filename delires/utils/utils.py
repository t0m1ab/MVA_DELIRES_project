import numpy as np
import csv
import json


def load_json(filename: str) -> dict:
    with open(filename, "r") as file:
        json_dict = json.load(file)
    return json_dict


def report_metrics(metrics, exp_path):
    img_names = list(metrics["PSNR"].keys())
    with open(exp_path, mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        fields = ["img", "PSNR", "data_fit", "std_over_image", "FID", "coverage", "LPIPS"]
        writer.writerow(fields)
        writer.writerow(["Overall"] + [np.mean(list(metrics[field].values())) for field in fields[1:]])
        for img in img_names:
            writer.writerow(
                [img]
                + [np.mean(metrics[field][img]) for field in fields[1:]]
                )
            
            
def archive_kwargs(kwargs, path):
    json.dump(kwargs, open(path, "w"), indent="\t")