import numpy as np
import csv
import json


def load_json(filename: str) -> dict:
    with open(filename, "r") as file:
        json_dict = json.load(file)
    return json_dict


def report_metrics(metrics, exp_path):
    with open(exp_path, mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        fields = ["img", "PSNR", "MSE", "Var_img", "Var_avg", "FID", "Cov"]
        writer.writerow(fields)
        writer.writerow(["Overall"] + [np.mean(metric) for metric in metrics])
        for i in range(len(metrics["PSNR"])):
            writer.writerow([i, np.mean(metrics["PSNR"][i])])
            
            
def archive_kwargs(kwargs, path):
    json.dump(kwargs, open(path, "w"), indent="\t")