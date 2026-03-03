import os
import glob
import json


if __name__ == "__main__":
    root_dir = "/storage4tb/PycharmProjects/GitHub/E2SIFT/output/vimeo_upsampled"
    json_path = "/storage4tb/PycharmProjects/GitHub/E2SIFT/data_processing/rename_log.json"

    data = json.load(open(json_path, "r"))

    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        os.rename(os.path.join(root_dir, folder), os.path.join(root_dir, data[f'{folder}.mp4'].replace('.mp4', '')))


        