import numpy as np
import json
import os

from PIL import Image
from matplotlib import pyplot as plt

from torch.utils.data import Dataset
import scipy.io
import cv2

import time


def quat2dcm(q):

    """ Computing direction cosine matrix from quaternion, adapted from PyNav. """

    # normalizing quaternion
    q = q/np.linalg.norm(q)

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    dcm = np.zeros((3, 3))

    dcm[0, 0] = 2 * q0 ** 2 - 1 + 2 * q1 ** 2
    dcm[1, 1] = 2 * q0 ** 2 - 1 + 2 * q2 ** 2
    dcm[2, 2] = 2 * q0 ** 2 - 1 + 2 * q3 ** 2

    dcm[0, 1] = 2 * q1 * q2 + 2 * q0 * q3
    dcm[0, 2] = 2 * q1 * q3 - 2 * q0 * q2

    dcm[1, 0] = 2 * q1 * q2 - 2 * q0 * q3
    dcm[1, 2] = 2 * q2 * q3 + 2 * q0 * q1

    dcm[2, 0] = 2 * q1 * q3 + 2 * q0 * q2
    dcm[2, 1] = 2 * q2 * q3 - 2 * q0 * q1

    return dcm

class Camera:
    def __init__(self,speed_root):

        """" Utility class for accessing camera parameters. """
        with open(os.path.join(speed_root, 'camera.json'), 'r') as f:
            camera_params = json.load(f)
        self.fx = camera_params['fx'] # focal length[m]
        self.fy = camera_params['fy'] # focal length[m]
        self.nu = camera_params['Nu'] # number of horizontal[pixels]
        self.nv = camera_params['Nv'] # number of vertical[pixels]
        self.ppx = camera_params['ppx'] # horizontal pixel pitch[m / pixel]
        self.ppy = camera_params['ppy'] # vertical pixel pitch[m / pixel]
        self.fpx = self.fx / self.ppx  # horizontal focal length[pixels]
        self.fpy = self.fy / self.ppy  # vertical focal length[pixels]
        self.k = camera_params['cameraMatrix']
        self.K = np.array(self.k) # cameraMatrix
        self.dcoef = camera_params['distCoeffs']


class PyTorchSatellitePoseEstimationDataset(Dataset):

    """ SPEED dataset that can be used with DataLoader for PyTorch training. """

    def __init__(self, split='train', speed_root='datasets/', transform_input=None, config=None):

        if split not in {'train', 'validation', 'sunlamp', 'lightbox', 'sunlamp_train', 'lightbox_train'}:
            
            raise ValueError('Invalid split, has to be either \'train\', \'validation\', \'sunlamp\' or \'lightbox\'')

        if split in {'train', 'validation'}:
            self.image_root = os.path.join(speed_root, 'synthetic', 'images')
            self.mask_root  = os.path.join(speed_root, 'synthetic', 'kptsmap_480')

            # We separate the if statement for train and val as we may need different training splits
            if split in {'train'}: 
                with open(os.path.join(speed_root, "synthetic", split + '.json'), 'r') as f:
                    label_list = json.load(f)

            if split in {'validation'}:
                with open(os.path.join(speed_root, "synthetic", split + '.json'), 'r') as f:
                    label_list = json.load(f)  

        else:
            self.image_root = os.path.join(speed_root, split, 'images')
            self.mask_root  = os.path.join(speed_root, split, 'kptsmap_480')
            
            if config["sunlamp_test"]:
                with open(os.path.join(speed_root, split, 'test.json'), 'r') as f:
                    label_list = json.load(f)
            else:
                with open(os.path.join(speed_root, split, 'train.json'), 'r') as f:
                    label_list = json.load(f)
        # Parse inputs
        self.sample_ids = [label['filename'] for label in label_list]
        self.train = (split == 'train') or  (not config["sunlamp_test"])
        self.validation = (split == 'validation')
        

       
        self.labels = {label['filename']: {'q': label['q_vbs2tango_true'], 'r': label['r_Vo2To_vbs_true']} for label in label_list}

        # Load assets
        kpts_mat  = scipy.io.loadmat(speed_root + "kpts.mat") 
        self.kpts = np.array(kpts_mat["corners"])  
        self.cam  = Camera(speed_root)              
        self.K    = self.cam.K

        self.K[0, :] *= ((config["cols"])/1920)
        self.K[1, :] *= ((config["rows"])/1200)    

        # Transforms for the tensors inputed to the network
        self.transform_input  = transform_input
        self.col_factor_input = ((config["cols"])/1920)
        self.row_factor_input = ((config["rows"])/1200)   

        self.split = split
        
        self.config = config
        

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        

        sample_id   = self.sample_ids[idx]
        
        # Read image        
        path_image  = os.path.join(self.image_root, sample_id)
        image  = cv2.imread(path_image)
        image_torch = self.transform_input(image)
        q0, r0 = self.labels[sample_id]['q'], self.labels[sample_id]['r']

        if self.train or self.validation:
            
            # Read heatmap                
            heatmap = np.zeros((self.config["rows"], self.config["cols"],11),dtype='uint8')

            target_name_02   = os.path.join(self.mask_root, sample_id.replace(".jpg","_02.png"))
            target_name_35   = os.path.join(self.mask_root, sample_id.replace(".jpg","_35.png"))
            target_name_69   = os.path.join(self.mask_root, sample_id.replace(".jpg","_69.png"))
            target_name_1011 = os.path.join(self.mask_root, sample_id.replace(".jpg","_1011.png"))

            img_02   = cv2.imread(target_name_02)
            img_35   = cv2.imread(target_name_35)
            img_69   = cv2.imread(target_name_69)
            img_1011 = cv2.imread(target_name_1011)

            heatmap[:,:,0:3]  = img_02
            heatmap[:,:,3:6]  = img_35
            heatmap[:,:,6:9]  = img_69
            heatmap[:,:,9:11] = img_1011[:,:,1:]

            kpts_vis = np.max(np.max(heatmap,axis=0),axis=0)
            heatmap_torch = self.transform_input(heatmap)
            
            # Get the pose
            q0, r0 = self.labels[sample_id]['q'], self.labels[sample_id]['r']

            ## Before computing the true key-point positions we need some transformations
            q  = quat2dcm(q0)
            r = np.expand_dims(np.array(r0),axis=1)
            r = q@(r)

            kpts_cam = q.T@(self.kpts+r)
            
            # Project key-points to the camera
            kpts_im = self.K@(kpts_cam/kpts_cam[2,:])
            kpts_im = np.transpose(kpts_im[:2,:])
        
        
        sample = dict()
        if (self.train or self.validation):
            
            sample["image"]     = image_torch
            sample["heatmap"]   = heatmap_torch
            sample["kpts"]      = kpts_im.astype(np.float32)
            sample["kpts_vis"]  = np.array(kpts_vis)

            sample["q0"]         = np.array(q0).astype(np.float32)  
            sample["r0"]         = np.array(r0).astype(np.float32)   
        else:
            sample["image"] = image_torch
            sample["y"]     = sample_id
            sample["q0"]    = np.array(q0).astype(np.float32)  
            sample["r0"]    = np.array(r0).astype(np.float32)   
        return sample

    