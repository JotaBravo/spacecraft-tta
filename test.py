import sys

import cv2
import numpy as np
from scipy.io import loadmat
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import kornia
import kornia.augmentation as K

from loaders import loader, loader_lb
from models import poseresnet
from src import speedscore
from src import utils



def dcm2quat(R):
    rot_1 = kornia.geometry.rotation_matrix_to_quaternion(R,order=kornia.geometry.conversions.QuaternionCoeffOrder.WXYZ).unsqueeze(0)
    rot_1 = rot_1[0].detach().cpu().numpy()
    return rot_1

mean        = 41.1280 
std         = 36.9064
#aug_intensity = K.AugmentationSequential(K.Normalize(mean=mean/255.0,std=std/255.0))
aug_intensity = K.AugmentationSequential(
    K.Normalize(mean=mean/255.0,std=std/255.0),
    K.RandomGaussianBlur((3, 3), (1.5, 1.5),p=1.0)
    )



with torch.no_grad():

    config_name = sys.argv[1]
    encoder_path  = sys.argv[2]
    decoder_path = sys.argv[3]
    config = utils.load_json(config_name)
    device = "cuda"
    
    tango_loader_lb = loader_lb.TangoVideoLoader(config)
    train_loader_lb = DataLoader(tango_loader_lb, batch_size=1, shuffle=False, num_workers=8)

    dist_coefs  = utils.get_coefs(config, device) # Distortion coefficients
    k_mat_input = utils.get_kmat_scaled(config, device) # Intrinsic matrix
    world_kpts = loadmat("dataset/roe1/kpts.mat")
    world_kpts = world_kpts["corners"].astype(np.float32).T

   
    encoder = poseresnet.get_encoder(50)
    encoder.to(device)
    encoder.load_state_dict(torch.load(encoder_path),strict=True)
    encoder.eval()
          
    kpt_decoder = poseresnet.get_decoder(50,final_layer=11)
    kpt_decoder.to(device)
    kpt_decoder.load_state_dict(torch.load(decoder_path),strict=True)
    kpt_decoder.eval()


    total_errors = []
    total_errors_pos = []
    total_errors_rot = []    





    npts = 8
    print("Number of keypoints: ", npts)

    errors = []
    errors_pos = []
    errors_rot = []
    for i, data in enumerate(tqdm(train_loader_lb, ncols=50)):

        img = aug_intensity(data["img"]).to(device)

        pose_gt = data["pose"]     
        quat_gt = data["quat"]
        t_gt= pose_gt[:,3:6]
        
        k_mat = utils.tensor_to_mat(k_mat_input[0]).copy()
        q_gt_b = np.expand_dims(utils.tensor_to_mat(quat_gt[0]),axis=1)
        t_gt_b = np.expand_dims(utils.tensor_to_mat(t_gt[0]),axis=1)

        feat_tgt = encoder(img)
        heatmap = kpt_decoder(feat_tgt)
        
        pred_mask = heatmap
        config["safe_margin_kpts"] = 1
        
        pred_mask = F.interpolate(heatmap, size=(img.shape[2],img.shape[3]), mode='bilinear', align_corners=True)

                 
        pred_kpts = utils.mask_to_points(pred_mask)
        pred_kpts = utils.tensor_to_mat(pred_kpts[0])

        flag_inside_x = np.bitwise_and(pred_kpts[:,0] > config["safe_margin_kpts"], pred_kpts[:,0] < (config["cols"]-config["safe_margin_kpts"]))
        flag_inside_y = np.bitwise_and(pred_kpts[:,1] > config["safe_margin_kpts"], pred_kpts[:,1] < (config["rows"]-config["safe_margin_kpts"]))
        flag_inside = np.bitwise_and(flag_inside_x, flag_inside_y)

        responses = np.max(np.max(utils.tensor_to_mat(heatmap[0]),axis=1),axis=1)


         
        index_max = np.argsort(-responses)

        world_kpts2  = world_kpts.copy()
        world_kpts2  = world_kpts2[index_max,:]
        pred_kpts = pred_kpts[index_max,:]
        flag_inside = flag_inside[index_max]
        
        world_kpts2 = world_kpts2[:npts:]
        pred_kpts = pred_kpts[:npts,:]        
        flag_inside = flag_inside[:npts]

        flag, rvecs_0, tvecs_0, inlier = cv2.solvePnPRansac(world_kpts2,pred_kpts, k_mat, distCoeffs=None, reprojectionError=5.0, flags=cv2.SOLVEPNP_EPNP)
        pos = tvecs_0
        rot = rvecs_0

        rot = cv2.Rodrigues(rot)[0]
        rot = torch.from_numpy(rot).unsqueeze(0).to(device)
        rot = dcm2quat(rot)  

        if pos[0] < -1:
            pos[0] = -1
        if pos[0] > 1:
            pos[0] = 1

        if pos[1] < -2:
            pos[1] = -2
        if pos[1] > 2:
            pos[1] = 2

        if pos[2] > 10:
            pos[2] = 10
        if pos[2] < 2:
            pos[2] = 2   

        error_0, error_pos_0, error_ori_0,_ = speedscore.speed_score(pos, rot, t_gt_b, q_gt_b, applyThresh=False, rotThresh=0.5, posThresh=0.005)

        errors.append(error_0)
        errors_rot.append(error_ori_0)
        errors_pos.append(error_pos_0)


        print("Total Errors: ",  np.mean(np.array(errors)))
        print("Rotation Errors: ", np.mean(np.array(errors_rot)))
        print("Position Errors: ", np.mean(np.array(errors_pos)))        