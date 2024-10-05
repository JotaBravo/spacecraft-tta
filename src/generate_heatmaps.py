import os
import json
import numpy as np
from scipy.io import loadmat
from loaders.speedplus import Camera
from utils import generate_map, load_config

from timeit import default_timer as timer
import cv2
from PIL import Image

import argparse
from multiprocessing import Pool
from multiprocessing import freeze_support
import tqdm

def run_multiprocessing(func, i, n_processors):
    pool = Pool(processes=n_processors)

    for _ in tqdm.tqdm(pool.imap_unordered(func, i), total=len(i)):
        pass
    #with Pool(processes=n_processors) as pool:
    #    return pool.map(func, i)
    
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', '--config', metavar='DIR', help='Path to the configuration', required=True)

args   = parser.parse_args()
config = load_config(args.cfg)

# Configuration parameters
path_speed  = config["root_dir"]
dataset_id  = config["split"]
if dataset_id == "train":
    dataset_id = "synthetic"
    
target_size = [config["rows"], config["cols"]]
gauss_std   = config["gauss_std"]
config["isloop"] = False

# Load ground-truth information
path_json  = os.path.join(path_speed, dataset_id, "train.json")
with open(path_json, 'r') as f:
    label_list = json.load(f)
sample_ids = [label['filename'] for label in label_list]
labels     = {label['filename']: {'q': label['q_vbs2tango_true'], 'r': label['r_Vo2To_vbs_true']} for label in label_list}

kpts = loadmat(path_speed + "kpts.mat")
kpts = np.array(kpts["corners"])
cam  = Camera(path_speed)

k_mat = cam.K
k_mat [0,:] *= ((target_size[0])/1920)
k_mat [1,:] *= ((target_size[1])/1200)     

path_save = os.path.join(path_speed, dataset_id, 'kptsmap')


def save_map(sample_id):

    heatmap = generate_map(labels, sample_id, target_size, gauss_std, k_mat, kpts).astype(np.uint8)
    
    # Save png
    target_name_02   = os.path.join(path_save, sample_id.replace(".jpg","_02.png"))
    target_name_35   = os.path.join(path_save, sample_id.replace(".jpg","_35.png"))
    target_name_69   = os.path.join(path_save, sample_id.replace(".jpg","_69.png"))
    target_name_1011 = os.path.join(path_save, sample_id.replace(".jpg","_1011.png"))
    
    cv2.imwrite(target_name_02,   heatmap[:,:,0:3])
    cv2.imwrite(target_name_35,   heatmap[:,:,3:6])
    cv2.imwrite(target_name_69,   heatmap[:,:,6:9])
    cv2.imwrite(target_name_1011, heatmap[:,:,8:])



def main():
 
    n_processors = config["num_workers"]
    run_multiprocessing(save_map, sample_ids, n_processors)
    
if __name__ == "__main__":
    freeze_support()   # required to use multiprocessinsg
    main()       