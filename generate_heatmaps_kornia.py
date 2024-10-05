from src import utils
from loaders import loader
from torch.utils.data import DataLoader
from tqdm import tqdm
from BPnP import BPnP
from kornia.geometry import render_gaussian2d
import torch
import cv2
import numpy as np
import os
from scipy.io import loadmat
from loaders import speedplus


config       = utils.load_json('configs/training/fixed/delta_1_ablation_pnp_render.json')
tango_loader = loader.TangoVideoLoader(config,mode="test")
tango_loader = DataLoader(tango_loader, 
                          batch_size=1, 
                          shuffle=False, 
                          num_workers=8,
                          drop_last=False)

k_mat         = utils.get_kmat_scaled(config, config["device"])[0]
points_world  = utils.get_world_kpts(config, config["device"])[0]


path_save = os.path.join(config["data_path"], "synthetic", "kptsmap")

if not os.path.exists(path_save):
    os.makedirs(path_save)


for i_iter, data in enumerate(tqdm(tango_loader)):
    

    pose = data["pose"].to(config["device"])
    
    q, r = data["quat"], data["t_tgt"]
    q = utils.tensor_to_mat(q[0])
    r = utils.tensor_to_mat(r[0])
    q = utils.quat2dcm(q)
    r = np.expand_dims(np.array(r),axis=1)
    r = q@(r)
    kpts = loadmat("/mnt/rhome/jbp/kelvins_esa/data/speedplus/" + "kpts.mat")
    kpts = np.array(kpts["corners"])

    cam = speedplus.Camera("/mnt/rhome/jbp/kelvins_esa/data/speedplus/")
    K = cam.K
    K[0, :] *= ((config["cols"])/1920)
    K[1, :] *= ((config["rows"])/1200)   
    
    

    tmp     = q.T@(kpts+r)
    kpts_im = K@(tmp/tmp[2,:])
    kpts_im = np.transpose(kpts_im[:2,:])    
    
    # check if cols and rows are ok or its the other way around
    flag_inside_x = np.bitwise_and(kpts_im[:,0] > config["safe_margin_kpts"], kpts_im[:,0] < (config["cols"]-config["safe_margin_kpts"]))
    flag_inside_y = np.bitwise_and(kpts_im[:,1] > config["safe_margin_kpts"], kpts_im[:,1] < (config["rows"]-config["safe_margin_kpts"]))
    flag_inside = np.bitwise_and(flag_inside_x, flag_inside_y)
            
            
    name = data["filename_tgt"][0]
    
    points = torch.from_numpy(kpts_im).to(config["device"]).unsqueeze(0)
    
    
    
    
    heatmap = render_gaussian2d(mean=points,
                                std=50*torch.ones_like(points,requires_grad=True),
                                size=config["target_size"],
                                normalized_coordinates=False)
    
    
    heatmap    = utils.tensor_to_mat(heatmap[0])
    heatmap    = np.transpose(heatmap,(1,2,0))
       
    for i in range(heatmap.shape[2]):
        if not flag_inside[i]:
            heatmap[:,:,i] = 0.0
        else:
            heatmap[:,:,i] = 255*(heatmap[:,:,i]/np.max(heatmap[:,:,i]))


    
    
    target_name_02   = os.path.join(path_save, name.replace(".jpg","_02.png"))
    target_name_35   = os.path.join(path_save, name.replace(".jpg","_35.png"))
    target_name_69   = os.path.join(path_save, name.replace(".jpg","_69.png"))
    target_name_1011 = os.path.join(path_save, name.replace(".jpg","_1011.png"))
    
    
    
    cv2.imwrite(target_name_02,   heatmap[:,:,0:3])
    cv2.imwrite(target_name_35,   heatmap[:,:,3:6])
    cv2.imwrite(target_name_69,   heatmap[:,:,6:9])
    cv2.imwrite(target_name_1011, heatmap[:,:,8:])

    np.save(target_name_02.replace("_02.png","_flag.npy"),   flag_inside)

    
