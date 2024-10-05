import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from src import utils
import json
from torchvision import transforms
from scipy.io import loadmat

def quat2euler(quat):

    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    ysqr = y * y
    t0 = -2.0 * (ysqr + z * z) + 1.0
    t1 = +2.0 * (x * y - w * z)
    t2 = -2.0 * (x * z + w * y)
    t3 = +2.0 * (y * z - w * x)
    t4 = -2.0 * (x * x + ysqr) + 1.0
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    roll = np.arctan2(t3, t4)
    pitch = np.arcsin(t2)
    yaw = np.arctan2(t1, t0)
    return np.array([roll, pitch, yaw])


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
        self.K = np.array(self.k,dtype=np.float32) # cameraMatrix
        self.dcoef = camera_params['distCoeffs']


class TangoVideoLoader(Dataset):

    def __init__(self,config, mode="train",input_transforms=[]):
        self.mode = mode
        self.config = config
        self.size = config["target_size"]
        self.data = utils.load_json(os.path.join(self.config['data_path'], 'train.json'))
        self.data = self.data['poses']

        self.mask_root = os.path.join(self.config['data_path'],"synthetic","kptsmap")
        
        self.filenames   = [data["filename"] for data in self.data]
        self.orientation = [data["q_vbs2tango_true"] for data in self.data]
        self.translation = [data["r_Vo2To_vbs_true"] for data in self.data]

        if input_transforms == -1:
            self.transforms = transforms.Compose([transforms.ToTensor(),transforms.Resize((self.size[0],self.size[1]))])
            self.skip_transforms = True
            
        else:
            self.transforms = input_transforms
            self.skip_transforms = False
        self.transforms_hm = transforms.Compose([transforms.ToTensor(),transforms.Resize((self.size[0],self.size[1]))])

        self.cam  = Camera(self.config['data_path'])              
        self.K    = self.cam.K

        self.K[0, :] *= ((self.size[0])/1920)
        self.K[1, :] *= ((self.size[1])/1200)    

        self.delta = config["delta"]
        kpts_mat  = loadmat("/mnt/rhome/jbp/kelvins_esa/data/speedplus/" + "kpts.mat")
        self.kpts = np.array(kpts_mat["corners"])   
        try:
           self.margin  = config["safe_margin_kpts"] 
        except:
            self.margin = 1.0
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        if self.mode == "train":
            idx = max(self.delta , idx)
            idx = min(len(self.filenames)-self.delta-1, idx)

        # Get data at t0
        filename_tgt    = self.filenames[idx]
        img = cv2.imread(os.path.join(self.config['data_path'],"synthetic","images", filename_tgt))
        if self.skip_transforms:
            img = self.transforms(img)
            
        #img = self.transforms(image=img)["image"]
        
        if self.mode == "train":
            heatmap = self.read_heatmap(filename_tgt)
            heatmap = self.transforms_hm(heatmap)

        orientation = quat2euler(np.array(self.orientation[idx]))
        translation_tgt = np.array(self.translation[idx])
        pose        = np.concatenate((orientation,translation_tgt),axis=0).astype(np.float32)
        quat = np.array(self.orientation[idx])
        
        
        dict_out = {"img":img, 
                    "pose":pose,
                    "K":self.K, 
                    "quat":quat,
                    "filename_tgt":filename_tgt,
                    "t_tgt":translation_tgt
                    }
        if self.mode == "train":
            ## Get data at t1
            filename_next    = self.filenames[idx+self.delta ]
            img_next = cv2.imread(os.path.join(self.config['data_path'],"synthetic","images", filename_next))
            
            if self.skip_transforms:
                img_next = self.transforms(img_next)
            #img_next = self.transforms(img_next)
            #img_next = self.transforms(image=img_next)["image"]
            if self.mode == "train":
                heatmap_next = self.read_heatmap(filename_next)
                heatmap_next = self.transforms_hm(heatmap_next)
#
            orientation = quat2euler(np.array(self.orientation[idx+self.delta]))
            translation = np.array(self.translation[idx+self.delta])
            pose_next   = np.concatenate((orientation,translation),axis=0).astype(np.float32)
            quat_next= np.array(self.orientation[idx+self.delta])            
            
            
            ## Get data at t-1
            filename_prev    = self.filenames[idx-self.delta ]
            img_prev = cv2.imread(os.path.join(self.config['data_path'],"synthetic","images", filename_prev))
            
            if self.skip_transforms:
                img_prev = self.transforms(img_prev)
            
            if not self.skip_transforms:
                dictaug = self.transforms(image=img,image0=img_next,image1=img_prev)
                img = dictaug["image"]
                img_next    = dictaug["image0"]
                img_prev = dictaug["image1"]

            if self.mode == "train":
                heatmap_prev = self.read_heatmap(filename_prev)
                heatmap_prev = self.transforms_hm(heatmap_prev)
#   
            orientation = quat2euler(np.array(self.orientation[idx-self.delta]))
            translation = np.array(self.translation[idx-self.delta])
            pose_prev   = np.concatenate((orientation,translation),axis=0).astype(np.float32)
            quat_prev = np.array(self.orientation[idx-self.delta])
            
            dict_out["img"] = img

            dict_out["img_next"] = img_next
            dict_out["pose_next"] = pose_next
            dict_out["img_prev"] = img_prev
            dict_out["pose_prev"] = pose_prev
            
            dict_out["quat_next"] = quat_next   
            dict_out["quat_prev"] = quat_prev
            

            # add heatmaps to dict
            
            dict_out["heatmap"] = heatmap
            dict_out["heatmap_next"] = heatmap_next
            dict_out["heatmap_prev"] = heatmap_prev

            dict_out["flag_outside"] = np.load(os.path.join(self.mask_root, filename_tgt[:-4] + "_flag.npy"))
            dict_out["flag_outside_next"] = np.load(os.path.join(self.mask_root, filename_next[:-4] + "_flag.npy"))
            dict_out["flag_outside_prev"] = np.load(os.path.join(self.mask_root, filename_prev[:-4] + "_flag.npy"))


            kpts = self.compute_kpts_locations(idx)
            dict_out["kpts"] = kpts
        #
            kpts_next = self.compute_kpts_locations(idx+self.delta)
            dict_out["kpts_next"] = kpts_next
        #
            kpts_prev = self.compute_kpts_locations(idx-self.delta)
            dict_out["kpts_prev"] = kpts_prev 
#
            ## this is because image is square
            #flag_outside_x = np.bitwise_and(kpts[:,0] > self.margin, kpts[:,0] < (self.size[0]-self.margin))
            #flag_outside_y = np.bitwise_and(kpts[:,1] > self.margin, kpts[:,1] < (self.size[1]-self.margin)) 
            #flag_outside = np.bitwise_and(flag_outside_x, flag_outside_y)
            #dict_out["flag_outside"] = flag_outside
            #
            #flag_outside_x = np.bitwise_and(kpts_next[:,0] > self.margin, kpts_next[:,0] < (self.size[0]-self.margin))
            #flag_outside_y = np.bitwise_and(kpts_next[:,1] > self.margin, kpts_next[:,1] < (self.size[1]-self.margin))
            #flag_outside_next = np.bitwise_and(flag_outside_x, flag_outside_y)
            #dict_out["flag_outside_next"] = flag_outside_next
            ##print(kpts_next[:,0],kpts_next[:,1])
            #            
            #flag_outside_x = np.bitwise_and(kpts_prev[:,0] > self.margin, kpts_prev[:,0] < (self.size[0]-self.margin))
            #flag_outside_y = np.bitwise_and(kpts_prev[:,1] > self.margin, kpts_prev[:,1] < (self.size[1]-self.margin))
            #
#
            #flag_outside_prev = np.bitwise_and(flag_outside_x, flag_outside_y)
            #dict_out["flag_outside_prev"] = flag_outside_prev
            
        return dict_out
        
        
        
    def read_heatmap(self,sample_id):
        
        heatmap = np.zeros((self.size[0], self.size[1],11),dtype='uint8')
        
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
        
        heatmap = heatmap
        return heatmap
    
    def compute_kpts_locations(self,sample_id):
        q0  = self.orientation[sample_id]
        r0  = self.translation[sample_id]
        
        q  = utils.quat2dcm(q0)
        r = np.expand_dims(np.array(r0),axis=1)
        r = q@(r)

        kpts_cam = q.T@(self.kpts+r)
        # Project key-points to the camera
        kpts_im = self.K@(kpts_cam/kpts_cam[2,:])
        kpts_im = np.transpose(kpts_im[:2,:])
        
        return kpts_im    