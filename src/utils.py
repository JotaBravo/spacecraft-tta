import json
import torch
from loaders import speedplus

# load json config
def load_json(config_file):

    with open(config_file) as f:
        config = json.load(f)
    return config


def dict_to_device(dictionary, device):
    for key, value in dictionary.items():
        dictionary[key] = dictionary[key].to(device)
    return dictionary

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)
    
    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    ## save grad_img_x as a scaled image    
    #grad_img_x_scaled = grad_img_x.clone()
    #grad_img_x_scaled = grad_img_x_scaled - torch.min(grad_img_x_scaled)
    #grad_img_x_scaled = grad_img_x_scaled / torch.max(grad_img_x_scaled)
    #grad_img_x_scaled = grad_img_x_scaled * 255.0
    #grad_img_x_scaled = grad_img_x_scaled.type(torch.uint8)
    #grad_img_x_scaled = grad_img_x_scaled[0]
    #grad_img_x_scaled = grad_img_x_scaled.permute(1, 2, 0)
    ## save image
    #import cv2
    #cv2.imwrite('grad_img_x.png', grad_img_x_scaled.cpu().numpy())   
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)


    ## save grad_img_x as a scaled image    
    #grad_img_x_scaled = grad_disp_x.clone()
    #grad_img_x_scaled = grad_img_x_scaled - torch.min(grad_img_x_scaled)
    #grad_img_x_scaled = grad_img_x_scaled / torch.max(grad_img_x_scaled)
    #grad_img_x_scaled = grad_img_x_scaled * 255.0
    #grad_img_x_scaled = grad_img_x_scaled.type(torch.uint8)
    #grad_img_x_scaled = grad_img_x_scaled[0]
    #grad_img_x_scaled = grad_img_x_scaled.permute(1, 2, 0)
    #cv2.imwrite('grad_disp_x.png', grad_img_x_scaled.cpu().numpy())   

    return grad_disp_x.mean() + grad_disp_y.mean()

def get_similarity_loss(disp, img,ssim_loss):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)
    print(grad_img_x.shape)


    out = ssim_loss(grad_img_x, grad_disp_x).mean() + ssim_loss(grad_img_y, grad_disp_y).mean()


    return out
import torch.nn as nn
class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
    
    
    
    
import numpy as np
import json
import cv2
import torch
import matplotlib.pyplot as plt
import os
import kornia
from kornia.geometry.conversions import QuaternionCoeffOrder
import scipy
import torch.nn.functional as F
from tqdm import tqdm
import shutil
def load_config(config_path):
    with open(config_path) as json_file:
        return json.load(json_file)    

def tensor_to_cvmat(tensor, BGR = True):
    mat = tensor.cpu().data.numpy().squeeze()

    if (tensor.shape[0]) == 3:
        mat = np.transpose(mat, (1, 2, 0))

    min_mat = np.min(mat)
    max_mat = np.max(mat)    
    mat = (mat-min_mat)/(max_mat-min_mat)
    out = (255*mat).astype(np.uint8)
    if BGR == True:
        out = cv2.cvtColor(out,cv2.COLOR_RGB2BGR)
    return out

def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d

def tensor_to_mat(tensor):
    tensor = tensor.cpu().data.numpy().squeeze()
    return tensor

def build_rt_mat(R,t): 
    a = torch.cat(R.shape[0]*[torch.Tensor([0,0,0,1]).unsqueeze(0)]).unsqueeze(1).to(R.get_device())
    b = torch.cat([R,t], dim = -1)
    c = torch.cat([b,a], dim = 1)
    return b

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

def project(q, r, K,dist):

        """ Projecting points to image frame to draw axes """

        # reference points in satellite frame for drawing axes
        p_axes = np.array([[0, 0, 0, 1],
                           [1, 0, 0, 1],
                           [0, 1, 0, 1],
                           [0, 0, 1, 1]])
        points_body = np.transpose(p_axes)

        # transformation to frame
        pose_mat = np.hstack((np.transpose(quat2dcm(q)), np.expand_dims(r, 1)))
        

        p_cam = np.dot(pose_mat, points_body)
        
        # getting homogeneous coordinates
        points_camera_frame = p_cam
        points_camera_frame[:,3] /=points_camera_frame[2,3]
  
        # projection to image plane
        #points_image_plane = K.dot(points_camera_frame)
        points_image_plane = points_camera_frame

        
        x0, y0 = (points_image_plane[0], points_image_plane[1])
        
        # apply distortion
        #dist = Camera.dcoef
        #dist = [1., 1., 1., 1, 1.]
        r2 = x0*x0 + y0*y0
        cdist = 1 + dist[0]*r2 + dist[1]*r2*r2 + dist[4]*r2*r2*r2
        x1  = x0*cdist + dist[2]*2*x0*y0 + dist[3]*(r2 + 2*x0*x0)
        y1  = y0*cdist + dist[2]*(r2 + 2*y0*y0) + dist[3]*2*x0*y0
        
        x = K[0,0]*x1 + K[0,2]
        y = K[1,1]*y1 + K[1,2]
        
        return x, y

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def draw_keypoints(image, keypoints, diameter=8, color = (255, 0, 0)):
    image = image.copy()
    for (x, y) in keypoints:
        cv2.circle(image, (int(x), int(y)), diameter, color, -1)
    return image

def heatmap_to_points(mask, temp=50.0):
    return kornia.geometry.subpix.spatial_soft_argmax2d(mask,temperature=torch.tensor(temp,requires_grad=True),normalized_coordinates=False)

def get_world_kpts(config,gpu):
    world_kpts = scipy.io.loadmat("/mnt/rhome/jbp/kelvins_esa/data/speedplus/"+ "kpts.mat")
    world_kpts = world_kpts["corners"].astype(np.float32).T
    world_kpts = torch.from_numpy(world_kpts).unsqueeze(0).repeat([config["batch_size"],1,1]).cuda(gpu).type(torch.float)
    return world_kpts

def load_corners(config,gpu):
    corners = scipy.io.loadmat('corners.mat')
    corners_out = {}
    corners_out["c1"] = torch.from_numpy(np.array(corners['corners1'])).unsqueeze(0).repeat([config["batch_size"],1,1]).cuda(gpu).type(torch.float)
    corners_out["c2"] = torch.from_numpy(np.array(corners['corners2'])).unsqueeze(0).repeat([config["batch_size"],1,1]).cuda(gpu).type(torch.float)
    corners_out["c3"] = torch.from_numpy(np.array(corners['corners3'])).unsqueeze(0).repeat([config["batch_size"],1,1]).cuda(gpu).type(torch.float)
    corners_out["c4"] = torch.from_numpy(np.array(corners['corners4'])).unsqueeze(0).repeat([config["batch_size"],1,1]).cuda(gpu).type(torch.float)
    corners_out["c5"] = torch.from_numpy(np.array(corners['corners5'])).unsqueeze(0).repeat([config["batch_size"],1,1]).cuda(gpu).type(torch.float)
    return corners_out

def get_coefs(config,gpu):
    cam = speedplus.Camera("/mnt/rhome/jbp/kelvins_esa/data/speedplus/")
    #coef = np.array(cam.dcoef)
    #coef = np.expand_dims(coef,axis=0)
    coef   = torch.from_numpy(np.array(cam.dcoef)).unsqueeze(0).repeat([config["batch_size"],1]).cuda(gpu).type(torch.float)
    return coef

def get_coefs_2(config,gpu):
    cam = speedplus.Camera("/mnt/rhome/jbp/kelvins_esa/data/speedplus/")
    #coef = np.array(cam.dcoef)
    #coef = np.expand_dims(coef,axis=0)
    coef   = torch.from_numpy(np.array(cam.dcoef)).unsqueeze(0).repeat([config["batch_size"],1]).cuda(gpu).type(torch.float)
    return coef

def get_coefs_test(config,gpu):
    cam = speedplus.Camera(config["data_path"])
    coef = np.array(cam.dcoef)
    coef = np.expand_dims(coef,axis=0)
    return coef


def get_kmat_scaled(config, gpu):

    cam = speedplus.Camera("/mnt/rhome/jbp/kelvins_esa/data/speedplus/")
    k = cam.K
    k[0, :] *= ((config["cols"])/1920)
    k[1, :] *= ((config["rows"])/1200)               
    k   = torch.from_numpy(k).unsqueeze(0).repeat([config["batch_size"],1,1]).cuda(gpu).type(torch.float)
    return k


def get_kmat_scaled_2(config, gpu):

    cam = speedplus.Camera(config["data_path"])
    k = cam.K
    k[0, :] *= ((config["cols"])/1920)
    k[1, :] *= ((config["rows"])/1200)               
    k   = torch.from_numpy(k).unsqueeze(0).repeat([config["batch_size"],1,1]).cuda(gpu).type(torch.float)
    return k


def get_kmat_scaled_test(config, gpu):

    cam = speedplus.Camera("/mnt/rhome/jbp/kelvins_esa/data/speedplus/")
    k = cam.K
    k[0, :] *= ((config["cols"])/1920)
    k[1, :] *= ((config["rows"])/1200)               
    k   = torch.from_numpy(k).unsqueeze(0).repeat([config["batch_size_test"],1,1]).cuda(gpu).type(torch.float)
    return k

def denormalize(grid):
    """Denormalize input grid from range [0, 1] to [-1, 1]
    Args:
        grid (Tensor): The grid to be denormalize, range [0, 1].

    Returns:
        Tensor: Denormalized grid, range [-1, 1].
    """

    return grid * 2.0 - 1.0

def point_sample(input, points, align_corners=True, **kwargs):
    add_dim = False
    if points.dim() == 3:
        add_dim = True
        points = points.unsqueeze(2)
    output = F.grid_sample(
        input, denormalize(points), align_corners=align_corners, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output     




def update_writer(writer,input_dict,iteration, config): # ppp + len(train_loader)*epoch

    with torch.no_grad():

        writer.add_scalar("Total Train Loss ",  input_dict["total_loss",0] + input_dict["total_loss",0], iteration)
        writer.add_scalar("Heatmap Loss ",      input_dict["loss_hm",0]      + input_dict["loss_hm" ,0], iteration)
        if config["activate_lpnp"]:                
            writer.add_scalar("PnP Loss ",          input_dict["loss_pnp",0]     + input_dict["loss_pnp" ,0], iteration)
        if config["activate_l3d"]:                
            writer.add_scalar("3D  Loss ",          input_dict["loss_3d",0]      + input_dict["loss_3d" ,0],  iteration)
            writer.add_scalar("Rotation  Loss ",    input_dict["rot_err",0]      + input_dict["rot_err" ,0],  iteration)
            writer.add_scalar("Translation  Loss ", input_dict["tra_err",0]      + input_dict["tra_err" ,0],  iteration)

        image = tensor_to_cvmat(input_dict["img_source"].sum(1,True)[0])
        image = draw_keypoints(image,tensor_to_mat(input_dict["kpts_gt"][0]),color=(0,255,0))
        image = cv2.resize(image,(200,200))

        hm_gt  = cv2.resize(tensor_to_cvmat(input_dict["heatmap_gt"].sum(1,True)[0]),(200,200))

        hm_c_0 = tensor_to_cvmat(input_dict["heatmap",0].sum(1,True)[0])

        if config["activate_lpnp"]:                
            hm_c_0 = draw_keypoints(hm_c_0,tensor_to_mat(input_dict["kpts_pnp",0][0]),color=(0,255,255))

        hm_c_0 = cv2.resize(hm_c_0,(200,200))

        hm_c_1 = tensor_to_cvmat(input_dict["heatmap",0].sum(1,True)[0])
        if config["activate_lpnp"]:                
            hm_c_1 = draw_keypoints(hm_c_1,tensor_to_mat(input_dict["kpts_pnp",0][0]),color=(0,255,255))
        hm_c_1 = cv2.resize(hm_c_1,(200,200))
        if config["activate_l3d"]:                

            kpts_3d_1 = BPnP.batch_project(input_dict["poses_3d",0], input_dict["kpts_world"] , input_dict["k_mat_input"])  
            kpts_3d_2 = BPnP.batch_project(input_dict["poses_3d",0], input_dict["kpts_world"] , input_dict["k_mat_input"])  


        depth_0 = tensor_to_cvmat(input_dict["depth",0].sum(1,True)[0])
        if config["activate_l3d"]:                
            depth_0 = draw_keypoints(depth_0,tensor_to_mat(kpts_3d_1[0]),color=(0,255,255))
        depth_0 = cv2.resize(depth_0,(200,200))
        depth_1 = tensor_to_cvmat(input_dict["depth",0].sum(1,True)[0])
        if config["activate_l3d"]:                
            depth_1 = draw_keypoints(depth_1,tensor_to_mat(kpts_3d_2[0]),color=(0,255,255))
        depth_1 = cv2.resize(depth_1,(200,200))                        
        a = np.vstack((image, hm_gt))
        b = np.vstack((hm_c_0, hm_c_1))
        c = np.vstack((depth_0,depth_1))
        d = np.hstack((a,b,c))
        writer.add_image("GT -- Heatmap + PnP -- Depth + 3D", d ,iteration,dataformats='HWC')



def eval_loop(val_loader ,aug_intensity_val, hourglass, kpts_world, k_mat_input, device, config):

    cam = speedplus.Camera(config["root_dir"])

    speed_total = 0
    speed_t_total = 0
    speed_r_total = 0
    kpts_world = tensor_to_mat(kpts_world[0])
    k_mat_input = tensor_to_mat(k_mat_input[0])
    for i_val, data in enumerate(tqdm(val_loader,ncols=50)):
    # Load data
        img_source = data["image"].to(device)
        img_source = aug_intensity_val(img_source)
        q_gt = tensor_to_mat(data["q0"][0])
        t_gt = tensor_to_mat(data["r0"][0])
        # Call the model
        out = hourglass(img_source) 
        pred_mask = out[1]["hm_c"]
        pred_mask = torch.nn.functional.interpolate(pred_mask, size=(img_source.shape[2],img_source.shape[3]), mode='bilinear', align_corners=False)
        pred_kpts  = heatmap_to_points(pred_mask) # Convert the heatmap to points
        pred_kpts  = tensor_to_mat(pred_kpts[0])
        aux, rvecs, t_est, inliers= cv2.solvePnPRansac(kpts_world, pred_kpts, k_mat_input, np.array(cam.dcoef), confidence=0.99,reprojectionError=1.0,flags=cv2.SOLVEPNP_EPNP)
        rvecs = torch.from_numpy(np.squeeze(rvecs)).unsqueeze(0).to(device)
        q_est = kornia.geometry.angle_axis_to_quaternion(rvecs, kornia.geometry.conversions.QuaternionCoeffOrder.WXYZ).unsqueeze(0)
        q_est = tensor_to_mat(q_est[0])
        speed, speed_t, speed_r = speedscore.speed_score(t_est, q_est, t_gt, q_gt, applyThresh=False)
        speed_total+=speed
        speed_t_total+=speed_t
        speed_r_total+=speed_r

    return speed_total/len(val_loader), speed_t_total/len(val_loader), speed_r_total/len(val_loader)

@torch.no_grad()
def eval_loop_simple(mse_loss, aug_intensity,val_loader, k_mat_input, dist_coefs, hourglass,device, config):
    hourglass.eval()
    val_loss = 0
    kmat_torch = get_kmat_scaled(config, device) # Intrinsic matrix

    for i, data in enumerate(tqdm(val_loader, ncols=50)):
         # Load the data
         img_source = data["image"].to(device)
         heatmap_gt = data["heatmap"].to(device)
         kpts_vis = data["visible_kpts"].to(device)

         img_source = aug_intensity(img_source)

         # Remove distortion
         img_source = kornia.geometry.undistort_image(img_source, kmat_torch, dist_coefs)    

         output = hourglass(img_source) #BxNKPTSxROWSxCOLS

         if config["activate_l3d"]:
            kpts = output[0]
            depth = output[1]
         else:
            kpts = output

         loss_hm = 0
         for level_id, level in enumerate([kpts]):
                # Interpolate network output to input resolution
                heatmap_pred = F.interpolate(level, size=(config["rows"],config["cols"]),
                                                            mode='bilinear',
                                                            align_corners=False)

                for batch_index in range(heatmap_pred.shape[0]):
                    # Get visible key-points
                    flag_vis = kpts_vis[batch_index]

                    # Get the batch predictions and ground-truth
                    pred_batch = heatmap_pred[batch_index,flag_vis,:,:]
                    gt_batch = heatmap_gt[batch_index,flag_vis,:,:]

                    loss_hm += mse_loss(pred_batch, 100*gt_batch)/10
         val_loss += loss_hm
    return val_loss/len(val_loader)

def mask_to_points(mask):
    img_points = kornia.geometry.subpix.spatial_soft_argmax2d(mask,temperature=torch.tensor(50.0,requires_grad=True),normalized_coordinates=False)

    #max_values  = torch.amax(mask, dim = [2,3] ,keepdim=True)
    #condition   = (mask ==max_values).type(torch.float)    
    #
    #indices     = torch.nonzero(condition,as_tuple=False)
    ### There might be more than one index that is max
    #indices,_      = torch.topk(indices,batch_size*npts,dim=0,sorted=False,largest=False)
    #img_points_0   = indices[:,2].view(mask.shape[0],mask.shape[1]).type(torch.float)
    #img_points_1   = indices[:,3].view(mask.shape[0],mask.shape[1]).type(torch.float)
    #img_points     = torch.stack((img_points_1,img_points_0),dim=2) # flip xy
    #print(img_points)
#
    #cv2.imwrite("kpts.jpg",draw_keypoints(tensor_to_cvmat(img[0]),tensor_to_mat(img_points[0]),diameter=5))
    #img_points = []
    #for mask_b in mask:
#
    #    max_indices  = torch.argmax(torch.flatten(mask_b,start_dim=1), dim = 1 ,keepdim=True) # torch.max returns the first maximum, argmax several
    #    x = max_indices / mask.shape[2] # (integer division in pytorch tensors is just `/` not `//`)
    #    y = max_indices % mask.shape[3]
    #    print(max_indices.requires_grad)
    #    img_points.append(torch.stack([y,x]))
    #img_points = torch.stack(img_points).squeeze().permute(0,2,1)ç


    return img_points
def generate_json_loop_ori(test_loader ,aug_intensity_test, hourglass, kpts_world, k_mat_input, device, config, path_json):

    cam = speedplus.Camera(config["root_dir"])
    npts= 8
    k_mat_input_tensor = k_mat_input
    kpts_world  = tensor_to_mat(kpts_world[0])
    k_mat_input = tensor_to_mat(k_mat_input[0])
    dist_coefs  = get_coefs(config, device) # Distortion coefficients
    dist_coefs = dist_coefs[0].unsqueeze(0)
    errors = []
    errors_pos = []
    errors_rot = []             
                
    with open(path_json, 'w') as fp:

        for i_val, data in enumerate(tqdm(test_loader,ncols=50)):
        # Load data
            img = data["image"].to(device)
            img = kornia.geometry.undistort_image(img, k_mat_input_tensor[0].unsqueeze(0), dist_coefs)

            img = aug_intensity_test(img)
            q_gt = data["q0"].to(device)
            t_gt = data["r0"].to(device)
            
            t_gt_b = np.expand_dims(tensor_to_mat(t_gt[0]),axis=1)
            q_gt_b = np.expand_dims(tensor_to_mat(q_gt[0]),axis=1)
            pred_mask_list = hourglass(img)
            pred_mask_0 = pred_mask_list[0]["hm_keypoint"]
            pred_mask_0  = torch.nn.functional.interpolate(pred_mask_0, size=(img.shape[2],img.shape[3]), mode='bilinear', align_corners=True)
            pred_kpts_0 = heatmap_to_points(pred_mask_0)

            pred_mask_1 = pred_mask_list[1]["hm_keypoint"]
            pred_mask_1  = torch.nn.functional.interpolate(pred_mask_1, size=(img.shape[2],img.shape[3]), mode='bilinear', align_corners=True)
            pred_kpts_1 = heatmap_to_points(pred_mask_1)
            pred_kpts_0 = tensor_to_mat(pred_kpts_0[0])
            pred_kpts_1 = tensor_to_mat(pred_kpts_1[0])
           # cv2.imwrite("kpts_0.jpg",draw_keypoints(tensor_to_cvmat(img[0]),pred_kpts_0))
           # cv2.imwrite("kpts_1.jpg",draw_keypoints(tensor_to_cvmat(img[0]),pred_kpts_1))

            responses_0 = np.max(np.max(tensor_to_mat(pred_mask_list[0]["hm_keypoint"]),axis=1),axis=1)
            responses_1 = np.max(np.max(tensor_to_mat(pred_mask_list[1]["hm_keypoint"]),axis=1),axis=1)

            index_max_0 = np.argsort(-responses_0)
            index_max_1 = np.argsort(-responses_1)

            world_kpts2 = kpts_world.copy()
            world_kpts2_1  = world_kpts2[index_max_1,:]
            world_kpts2_0  = world_kpts2[index_max_0,:]

            pred_kpts_0 = pred_kpts_0[index_max_0,:]
            pred_kpts_1 = pred_kpts_1[index_max_1,:]

            world_kpts2_0   = world_kpts2_0[:npts,:]
            world_kpts2_1   = world_kpts2_1[:npts,:]

            pred_kpts_0     = pred_kpts_0[:npts,:]        
            pred_kpts_1     = pred_kpts_1[:npts,:]

            total_rp0 = responses_0[index_max_0]
            total_rp0 = np.sum(total_rp0[:npts])

            total_rp1 = responses_1[index_max_1]
            total_rp1 = np.sum(total_rp1[:npts])

            aux_0, rvecs_0, tvecs_0, inliers_0= cv2.solvePnPRansac(world_kpts2_0, pred_kpts_0,k_mat_input,distCoeffs=None,confidence=0.99,reprojectionError=2.0,flags=cv2.SOLVEPNP_EPNP)
            aux_1, rvecs_1, tvecs_1, inliers_1= cv2.solvePnPRansac(world_kpts2_1, pred_kpts_1,k_mat_input,distCoeffs=None,confidence=0.99,reprojectionError=2.0,flags=cv2.SOLVEPNP_EPNP)


            rot_0 = cv2.Rodrigues(rvecs_0)[0]
            rot_1 = cv2.Rodrigues(rvecs_1)[0]
            
            error_pnp_og_vs_rot = compute_error(rot_0,tvecs_0,rot_1,tvecs_1)

            if aux_0 and aux_1 and error_pnp_og_vs_rot[0] < 0.1:
                if total_rp0 > total_rp1:
                    out_tvec = tvecs_0
                    out_rvec = rvecs_0
                else:
                    out_tvec = tvecs_1
                    out_rvec = rvecs_1
                q_est = torch.from_numpy(np.squeeze(out_rvec)).unsqueeze(0).to(device)
                q_est = kornia.geometry.angle_axis_to_quaternion(q_est,kornia.geometry.conversions.QuaternionCoeffOrder.WXYZ).unsqueeze(0)
                q_est = tensor_to_mat(q_est[0])
                t_est = out_tvec
                sample = {}
                sample["q_vbs2tango_true"] = q_est.tolist()
                sample["r_Vo2To_vbs_true"] = np.squeeze(out_tvec.T).tolist()
                sample["filename"] = data["y"][0]              

                error_0, error_pos_0, error_ori_0 = speedscore.speed_score(out_tvec, q_est, t_gt_b, q_gt_b, applyThresh=False, rotThresh=0.5, posThresh=0.005)
                errors.append(error_0)
                errors_rot.append(error_ori_0)
                errors_pos.append(error_pos_0)
                json.dump(sample, fp)


        print("total samples", len(errors))
        print("Mean error: ", np.mean(errors))
        print("mean error rot: ", np.mean(errors_rot))
        print("mean error pos: ", np.mean(errors_pos))
        print("------------------------")
def generate_json_loop(test_loader ,aug_intensity, model, kpts_world, k_mat_input, device, config, path_json):

    cam = speedplus.Camera(config["root_dir"])

    world_kpts  = tensor_to_mat(kpts_world[0])
    k_mat_input = tensor_to_mat(k_mat_input[0])
    dist_coefs  = get_coefs_test(config, device) # Distortion coefficients
    target_size  = (config["rows"],config["cols"])
    nkpts = 8
    reproj_error = 2.0
    angle_rotation = 15.0
    k_mat_torch = get_kmat_scaled_test(config, device) # Intrinsic matrix
    k_mat_numpy = tensor_to_mat(k_mat_torch).copy()

    with open(path_json, 'w') as fp:

        for i_val, data in enumerate(tqdm(test_loader,ncols=50)):
            
            img_source = data["image"].to(device)
            img_source = kornia.geometry.undistort_image(img_source, k_mat_torch, dist_coefs)

            # Evaluate model
            img_source = aug_intensity(img_source)
            
            
            # Get the predicted pose
            heatmap, depth = model(img_source)
            heatmap = F.interpolate(heatmap,size=target_size,mode='bilinear',align_corners=True)
            depth   = F.interpolate(depth, size=target_size,mode='bilinear' ,align_corners=True)

            kpts, response = heatmap_to_keypoints(heatmap[0],filter_edges=False)
            depth_sampled  = sample_depth(depth[0], kpts)

            pose_pnp, flag = estimate_pnp(world_kpts, kpts, response, k_mat_numpy, nkpts=nkpts, reproj_error=reproj_error) #Rot,Pos

            # Get the predicted pose for the flipped image
            angle_rotation_in = np.random.rand(1)*float(angle_rotation) + 5.0
            img_source = kornia.geometry.transform.rotate(img_source, torch.tensor(angle_rotation_in).to(device).type(torch.float),padding_mode="border")
            heatmap, depth = model(img_source)
            heatmap = kornia.geometry.transform.rotate(heatmap, torch.tensor(-angle_rotation_in).to(device).type(torch.float),padding_mode="border")
            img_source = kornia.geometry.transform.rotate(img_source, torch.tensor(-angle_rotation_in).to(device).type(torch.float),padding_mode="border")

            heatmap = F.interpolate(heatmap,size=target_size,mode='bilinear',align_corners=True)
            depth   = F.interpolate(depth, size=target_size,mode='bilinear' ,align_corners=True)
            kpts, response = heatmap_to_keypoints(heatmap[0],filter_edges=False)
            depth_sampled  = sample_depth(depth[0], kpts)
            pose_pnp_rot, flag_rot = estimate_pnp(world_kpts, kpts, response, k_mat_numpy, nkpts=nkpts, reproj_error=reproj_error)



            rot_aug = torch.from_numpy(pose_pnp_rot[0]).unsqueeze(0).to("cuda:0")
            rot_aug = dcm2quat(rot_aug)  
            error_pnp_og_vs_rot = compute_error(pose_pnp[0], pose_pnp[1], rot_aug, pose_pnp_rot[1])


            
            if np.bitwise_and(flag[0], flag_rot[0]) and error_pnp_og_vs_rot[0] < 0.2: 

                q_est = torch.from_numpy(pose_pnp[0]).unsqueeze(0).to("cuda:0")
                q_est = dcm2quat(q_est)  
                q_est = np.squeeze(q_est)
                t_est = pose_pnp[1]
                sample = {}
                sample["q_vbs2tango_true"] = q_est.tolist()
                sample["r_Vo2To_vbs_true"] = np.squeeze(t_est.T).tolist()
                sample["filename"] = data["y"][0]              

                json.dump(sample, fp)



import time
from scipy.io import savemat
def test_loop(val_loader ,aug_intensity_val, hourglass, kpts_world, k_mat_input, device, config):

    cam = speedplus.Camera(config["root_dir"])

    speed_total = 0
    speed_t_total = 0
    speed_r_total = 0
    kpts_world = tensor_to_mat(kpts_world[0])
    k_mat_input = tensor_to_mat(k_mat_input[0])
    speed_vals = []
    speed_t_vals = []
    speed_r_vals = []

    for i_val, data in enumerate(tqdm(val_loader,ncols=50)):
        img  = data["image"].to(device)

        q_gt = tensor_to_mat(data["q0"][0])
        t_gt = tensor_to_mat(data["r0"][0])

        img = aug_intensity_val(img)
        pred_mask_list = hourglass(img)
        pred_mask_0 = pred_mask_list[0]["hm_c"]
        pred_mask_0  = torch.nn.functional.interpolate(pred_mask_0, size=(img.shape[2],img.shape[3]), mode='bilinear', align_corners=True)
        pred_kpts_0 = heatmap_to_points(pred_mask_0)
        pred_mask_1 = pred_mask_list[1]["hm_c"]
        pred_mask_1  = torch.nn.functional.interpolate(pred_mask_1, size=(img.shape[2],img.shape[3]), mode='bilinear', align_corners=True)
        pred_kpts_1 = heatmap_to_points(pred_mask_1)


        pred_kpts_0 = tensor_to_mat(pred_kpts_0[0])
        pred_kpts_1 = tensor_to_mat(pred_kpts_1[0])

        
        #cv2.imwrite( "test_mse/ktpts" + str(i_val) + ".jpg" ,np.hstack((draw_keypoints(tensor_to_cvmat(img[0]),pred_kpts_0),draw_keypoints(tensor_to_cvmat(img[0]),pred_kpts_1))))
        responses_0 = np.max(np.max(tensor_to_mat(pred_mask_list[0]["hm_c"]),axis=1),axis=1)
        responses_1 = np.max(np.max(tensor_to_mat(pred_mask_list[1]["hm_c"]),axis=1),axis=1)
        index_max_0 = np.argsort(-responses_0)
        index_max_1 = np.argsort(-responses_1)
        world_kpts2 = kpts_world.copy()
        world_kpts2_1  = world_kpts2[index_max_1,:]
        world_kpts2_0  = world_kpts2[index_max_0,:]
        pred_kpts_0 = pred_kpts_0[index_max_0,:]
        pred_kpts_1 = pred_kpts_1[index_max_1,:]
        npts = 7
        world_kpts2_0   = world_kpts2_0[:npts,:]
        world_kpts2_1   = world_kpts2_1[:npts,:]
        pred_kpts_0     = pred_kpts_0[:npts,:]        
        pred_kpts_1     = pred_kpts_1[:npts,:]
        total_rp0 = responses_0[index_max_0]
        total_rp0 = np.sum(total_rp0[:npts])
        total_rp1 = responses_1[index_max_0]

        total_rp1 = np.sum(total_rp1[:npts])
        aux_0, rvecs_0, tvecs_0, inliers= cv2.solvePnPRansac(world_kpts2_0, pred_kpts_0, k_mat_input, np.array(cam.dcoef),confidence=0.99,reprojectionError=1.0,flags=cv2.USAC_MAGSAC)
        aux_1, rvecs_1, tvecs_1, inliers= cv2.solvePnPRansac(world_kpts2_1, pred_kpts_1, k_mat_input, np.array(cam.dcoef),confidence=0.99,reprojectionError=1.0,flags=cv2.USAC_MAGSAC)
  
        if total_rp0 > total_rp1:
            out_tvec = tvecs_0
            out_rvec = rvecs_0
        else:
            out_tvec = tvecs_1
            out_rvec = rvecs_1
        #out_tvec = tvecs_1
        #out_rvec = rvecs_1

        q_est = torch.from_numpy(np.squeeze(out_rvec)).unsqueeze(0).to(device)
        q_est = kornia.geometry.angle_axis_to_quaternion(q_est,kornia.geometry.conversions.QuaternionCoeffOrder.WXYZ).unsqueeze(0)
        q_est = tensor_to_mat(q_est[0])
        t_est = out_tvec
        #aux, rvecs, t_est, inliers= cv2.solvePnPRansac(kpts_world, pred_kpts, k_mat_input, np.array(cam.dcoef), confidence=0.99,reprojectionError=1.0,flags=cv2.SOLVEPNP_EPNP)
        #rvecs = torch.from_numpy(np.squeeze(rvecs)).unsqueeze(0).to(device)
        #q_est = kornia.geometry.angle_axis_to_quaternion(rvecs, kornia.geometry.conversions.QuaternionCoeffOrder.WXYZ).unsqueeze(0)
        #q_est = tensor_to_mat(q_est[0])



        speed, speed_t, speed_r = speedscore.speed_score(t_est, q_est, t_gt, q_gt, applyThresh=False)
        speed_total+=speed
        speed_t_total+=speed_t
        speed_r_total+=speed_r
        speed_vals.append(speed)
        speed_t_vals.append(speed_t)
        speed_r_vals.append(speed_r)        
     #   vprint(speed)
    #time.sleep(1)

    savemat('lb_all.mat', mdict={'speed_all': speed_vals,'speed_t_all': speed_t_vals,'speed_r_all': speed_r_vals})
    return speed_total/len(val_loader), speed_t_total/len(val_loader), speed_r_total/len(val_loader)    





def format_and_copy_json(previous_config):
    config = load_config(previous_config)
    print("Changing the format to the labels")

    input_file  = os.path.join(config["path_results"], previous_config,"train.json")
    output_file = os.path.join(config["path_results"], previous_config,"train_clean.json")
    print("\t --input file: ", input_file)
    print("\t --output file: ", output_file)


    # Read in the file
    with open(input_file, 'r') as file :
        filedata = file.read()

    # Replace the target string
    filedata = filedata.replace('}', '},')
    filedata = "[" + filedata[:-1] + "]"

    # Write the file out again
    with open(output_file, 'w') as file:
        file.write(filedata)

    src = os.path.join(output_file)
    
    dst = os.path.join(config["root_dir"],"sunlamp","train.json")
    print("coping labels to: ", dst)
    shutil.copyfile(src, dst)        
    
    

def head_to_kpts(head):
    
    max_per_channel = np.max(np.max(head,axis=2),axis=1)
    kpts = np.zeros((head.shape[0],2))
    for i, max_ch in enumerate(max_per_channel):
        kpt = np.where(head[i] == max_ch)
        
        
        if len(kpt[0])>1:
            a = kpt[0][0]
        else:
            a = kpt[0]    
        if len(kpt[0])>1:
            b = kpt[0][1]
        else:
            b = kpt[1]                       
        kpts[i,1] = a
        kpts[i,0] = b 
    return kpts, max_per_channel


def heatmap_to_keypoints(heatmap, filter_edges=False):
    heatmap = tensor_to_mat(heatmap)
    kpts, response = head_to_kpts(heatmap)
    
    if filter_edges:
        min_index = np.bitwise_and.reduce(kpts > 480*0.1,axis=1)
        max_index = np.bitwise_and.reduce(kpts < 480*0.9,axis=1)
    
        idex_keep = np.bitwise_and(min_index,max_index)
        kpts = kpts[idex_keep]
        response = response[idex_keep]

    return kpts, response

def sample_depth(depth, kpts, scale=1.0):
    kpts = kpts.astype(int)
    kpts = kpts*scale
    kpts = kpts.astype(int)
    
    depth = tensor_to_mat(depth)  
    
    depths_out = []
    for i, kpt in enumerate(kpts):
        depths_out.append(depth[i,kpt[0],kpt[1]])

    return np.array(depths_out)

def mask_from_pose(img, q_batch, tvecs, k_mat_torch,c1,c2,c3,c4,c5,device):


    r_batch = torch.from_numpy(np.squeeze(tvecs.T)).unsqueeze(0).to(device).unsqueeze(2).type(torch.float)
    gen_mask = create_mask(img, q_batch, r_batch, k_mat_torch, c1, c2, c3, c4, c5)
    mask_save  = tensor_to_cvmat(gen_mask[0])

    return mask_save


def create_mask(img2, pred_rot, pred_trans, k, c1, c2, c3, c4, c5):
    # square images
    k1    = torch.clip(project_kpts(c1,pred_rot,pred_trans, k),min=0.1,max=img2.shape[2]-1)
    k2    = torch.clip(project_kpts(c2,pred_rot,pred_trans, k),min=0.1,max=img2.shape[2]-1)
    k3    = torch.clip(project_kpts(c3,pred_rot,pred_trans, k),min=0.1,max=img2.shape[2]-1)
    k4    = torch.clip(project_kpts(c4,pred_rot,pred_trans, k),min=0.1,max=img2.shape[2]-1)
    k5    = torch.clip(project_kpts(c5,pred_rot,pred_trans, k),min=0.1,max=img2.shape[2]-1)
    #k1    = project_kpts(c1, pred_rot, pred_trans, k)

    #k2    = project_kpts(c2, pred_rot,pred_trans, k)
    #k3    = project_kpts(c3, pred_rot,pred_trans, k)
    #k4    = project_kpts(c4, pred_rot,pred_trans, k)
    #k5    = project_kpts(c5, pred_rot,pred_trans, k)
    img3= torch.zeros_like(img2)
    color = torch.tensor([[1.0, 1.0, 1.0]],requires_grad=False).to("cuda")

    color = color.repeat(img2.shape[0],1)

    mask1    = kornia.utils.draw_convex_polygon(img3, k1, color)
   
    mask2    = kornia.utils.draw_convex_polygon(img3, k2, color)
    mask3    = kornia.utils.draw_convex_polygon(img3, k3, color)
    mask4    = kornia.utils.draw_convex_polygon(img3, k4, color)
    mask5    = kornia.utils.draw_convex_polygon(img3, k5, color)
    mask_out = mask1 + mask2 + mask3 + mask4 + mask5
    mask_out = torch.clip(mask_out,min = 0,max = 1.0)
    return mask_out.mean(1,True)


def project_kpts(kpts,pred_rot,r0, k):

    #pred_rot   = q0.unsqueeze(1).unsqueeze(0)
    pred_trans = r0

    pred_rot   = kornia.geometry.quaternion_to_rotation_matrix(pred_rot,QuaternionCoeffOrder.WXYZ).type(torch.float)

    pred_rot   = pred_rot.permute([0,2,1]) #for some reason this is needed compared to MATLAB    



    # let's use the same coordinate framework as in the ground-truth
    # 1 - r = q*r



    pred_trans = torch.bmm(pred_rot, pred_trans) #Bx1x3
    # 2 - c = q'*(corners+r)
    kpts = (kpts + pred_trans)
    pred_rot = pred_rot.permute([0 ,2 ,1])

    
    kpts = torch.bmm(pred_rot, kpts) #Bx1x3

    kpts = kpts/(kpts[:,2,:].unsqueeze(1).repeat([1, 3, 1]))


    kpts = torch.bmm(k,kpts)
    kpts = (kpts[:,:2,:]).permute([0,2,1])
    
    return kpts



def draw_heatmap(heatmap,name, plot_point = False):
    ch_list = []    
    for ch in heatmap:
        a_ = tensor_to_mat(ch)
        a = tensor_to_cvmat(ch)
        
        kpt = np.where(a_ == np.max(a_))
        a_text = cv2.putText(a, str(np.max(a_)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        if plot_point:
            a_text = cv2.circle(a_text, (int(kpt[1]), int(kpt[0])), 8, (0,255,0), -1)
        ch_list.append(a_text)
        
    ch_list.append(tensor_to_cvmat(heatmap.sum(0,True)))
    A = np.hstack(ch_list[0:4])
    B = np.hstack(ch_list[4:8])
    C = np.hstack(ch_list[8:12])
    E = np.vstack((A,B,C))
    cv2.imwrite(name,E)
    
    
def draw_image_kpts(img, heatmap, name):
    img = tensor_to_cvmat(img)     
    heatmap = tensor_to_mat(heatmap)
    
    kpts, resps = head_to_kpts(heatmap)      
    img = draw_keypoints(img,kpts,4,(0,255,0))
    cv2.imwrite(name,img)
    

def dcm2quat(R):
    rot_1 = torch.from_numpy(R).unsqueeze(0).to(device)
    rot_1 = kornia.geometry.rotation_matrix_to_quaternion(R,order=kornia.geometry.conversions.QuaternionCoeffOrder.WXYZ).unsqueeze(0)
    rot_1 = rot_1[0].detach().cpu().numpy()
    return rot_1


from scipy.io import loadmat,savemat

def load_corners():
    corners = loadmat('corners.mat')
    device = "cuda:0"
    c1 = torch.from_numpy(np.array(corners['corners1'])).unsqueeze(0).repeat([1,1,1]).to(device).type(torch.float)
    c2 = torch.from_numpy(np.array(corners['corners2'])).unsqueeze(0).repeat([1,1,1]).to(device).type(torch.float)
    c3 = torch.from_numpy(np.array(corners['corners3'])).unsqueeze(0).repeat([1,1,1]).to(device).type(torch.float)
    c4 = torch.from_numpy(np.array(corners['corners4'])).unsqueeze(0).repeat([1,1,1]).to(device).type(torch.float)
    c5 = torch.from_numpy(np.array(corners['corners5'])).unsqueeze(0).repeat([1,1,1]).to(device).type(torch.float)
    
    return [c1,c2,c3,c4,c5]
def get_worldkpts(config):
    world_kpts = loadmat(config["root_dir"] + "kpts.mat")
    world_kpts = world_kpts["corners"].astype(np.float32).T

    return world_kpts



def estimate_pnp(kpts_world,kpts, response, k_mat,nkpts=11,reproj_error=2.0):
    
    index = np.argsort(-response)
    kpts_world = kpts_world[index]           
    kpts = kpts[index]    
    
    flag, rot, pos, inlier = cv2.solvePnPRansac(kpts_world[:nkpts], kpts[:nkpts], k_mat, None, confidence=0.999, reprojectionError=reproj_error, flags=cv2.SOLVEPNP_EPNP)
    
    rot = cv2.Rodrigues(rot)[0]

    return [rot, pos], [flag, inlier]



def compute_error(rot,pos,q_gt_b,t_gt_b):
    
    rot = torch.from_numpy(rot).unsqueeze(0).to("cuda:0")
    rot = dcm2quat(rot)  
    
    
    q_gt_b = torch.from_numpy(q_gt_b).unsqueeze(0).to("cuda:0")
    q_gt_b = dcm2quat(q_gt_b)  
    
    
    error, error_pos, error_ori = speedscore.speed_score(pos, rot, t_gt_b, q_gt_b, applyThresh=False, rotThresh=0.5, posThresh=0.005)

    return [error, error_ori, error_pos]


def draw_pose(img,rot,pos,k_mat, name):
    
    c = load_corners()
    rot = torch.from_numpy(rot).unsqueeze(0).to("cuda")
    rot = kornia.geometry.rotation_matrix_to_quaternion(rot,order=kornia.geometry.conversions.QuaternionCoeffOrder.WXYZ).unsqueeze(0)
    mask  = mask_from_pose(img, rot, pos, k_mat, c[0],c[1],c[2],c[3],c[4],"cuda")
    img = tensor_to_cvmat(img[0])
    cv2.imwrite(name,0.5*img +0.5*mask)

def draw_labelmap(img, pt, sigma, type='Gaussian'):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    
    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= 0.95*img.shape[1] or ul[1] >= 0.95*img.shape[0] or br[0] < 10 or br[1] < 10):
        # If not, just return the image as is
        return img, False
    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)
    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])
    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img, True


def generate_map(labels, sample_id, target_size, gaussian_std, K, kpts):
    
    q, r = labels[sample_id]['q'], labels[sample_id]['r']
    q = quat2dcm(q)
    r = np.expand_dims(np.array(r),axis=1)
    r = q@(r)

    tmp     = q.T@(kpts+r)
    kpts_im = K@(tmp/tmp[2,:])
    kpts_im = np.transpose(kpts_im[:2,:])

    pil_drawkpts = np.zeros((target_size[0], target_size[1], 11),dtype='float32')
    
    for i, pt in enumerate(kpts_im):
        aux, vis = draw_labelmap(pil_drawkpts[:,:,i], pt, gaussian_std, type='Gaussian')
        
        if vis:
            pil_drawkpts[:,:,i]  = (aux/np.max(aux))*255.0

    return pil_drawkpts


def dict_to_device(dictionary, device):
    for key, value in dictionary.items():
        if "filename" not in key:
            dictionary[key] = dictionary[key].to(device)
    return dictionary



def unsup(hm, num_hm=11):
    diff = 0
    d = hm.view(hm.shape[0]*num_hm, hm.shape[2],hm.shape[3])
    
    logits = d.flatten(start_dim=1)
    prob = torch.nn.functional.softmax(logits, dim=1)+1e-30
    diff = (-prob*torch.log(prob)).sum(1)
    return diff.mean()


def augmentate_images(data, augmentation):
    for key in data:
        if "img" in key:
            data[key] = augmentation(data[key])
    return data

def forward_pass(img, model, config, return_depth=False):
    output = model(img)[0]
    heatmap = output["hm_keypoint"]
    heatmap = F.interpolate(heatmap, size=config["target_size"], mode='bilinear', align_corners=True)
    
    if return_depth:
        depth = output["depth"]
        depth = F.interpolate(depth, size=config["target_size"], mode='bilinear', align_corners=True)
        return heatmap, depth
    else:
        return heatmap, output



def rt_to_pose(rt):
    R = kornia.geometry.conversions.angle_axis_to_rotation_matrix(rt[:,0:3])

    t = rt[:,3:6].unsqueeze(2)
    pose = build_rt_mat(R, t)
    return pose


def relative_rt(rt_target, rt_source):
    
    R_tgt = kornia.geometry.conversions.angle_axis_to_rotation_matrix(rt_target[:,0:3])
    t_tgt = rt_target[:,3:6]
          
    R_source = kornia.geometry.conversions.angle_axis_to_rotation_matrix(rt_source[:,0:3])
    t_source = rt_source[:,3:6]
    
    # Compute relative poses
    R_next_tgt  = torch.matmul(R_tgt, R_source.transpose(2,1))
    t_next_tgt  = -torch.matmul(R_next_tgt, t_source.unsqueeze(2)) + t_tgt.unsqueeze(2) 
    
    return build_rt_mat(R_next_tgt, t_next_tgt)
       
    
def relative_poses(heatmap_target, heatmap_source, kpts_world, k_mat_input, pnp_fast):
    
    # Derive pose via PnP
    kpts_pred = heatmap_to_points(heatmap_target, temp=10.0)
    rt_target = pnp_fast(kpts_pred, kpts_world[0], k_mat_input[0]) # Bx6 [rot,pose]
    R_tgt = kornia.geometry.conversions.angle_axis_to_rotation_matrix(rt_target[:,0:3])
    t_tgt = rt_target[:,3:6]
        
        
    # Derive pose via PnP
    kpts_pred = heatmap_to_points(heatmap_source, temp=10.0)
    rt_source = pnp_fast(kpts_pred, kpts_world[0], k_mat_input[0]) # Bx6 [rot,pose]
    R_source = kornia.geometry.conversions.angle_axis_to_rotation_matrix(rt_source[:,0:3])
    t_source = rt_source[:,3:6]
    
    
    # Compute relative poses
    R_next_tgt  = torch.matmul(R_tgt, R_source.transpose(2,1))
    t_next_tgt  = -torch.matmul(R_next_tgt, t_source.unsqueeze(2)) + t_source.unsqueeze(2)

    return build_rt_mat(R_next_tgt, t_next_tgt), R_tgt, t_tgt

def warp_image(img, t_tgt, rt_mat, data):
    

    depth_scale = torch.ones_like(img[:,0:1,:,:])#*t_tgt[:,2:3].unsqueeze(3)
    warped_img = kornia.geometry.warp_frame_depth(img, depth_scale, rt_mat, data["K"])
    
    return warped_img



def rob_loss(hm,num_hm=11):
    
    #d = hm.view(hm.shape[0], hm.shape[1,] hm.shape[2],hm.shape[3])
    #print(d.shape)
    hm_pool = F.avg_pool2d(hm, 3    , stride=1,padding=[1,1])
    #print(hm_pool.shape)
    hm_pool_flat = torch.flatten(hm_pool,start_dim=2) # Bxhmxwindows
    #print(hm_pool_flat.shape)

    # Extract the
    prob = F.softmax(hm_pool_flat, dim=2).clamp(min=1e-16)


    diff = -(prob*torch.log(prob)).sum()/prob.sum()
   
    return diff



def dcm2quat(R):
    #rot_1 = torch.from_numpy(R).unsqueeze(0).to(device)
    rot_1 = kornia.geometry.rotation_matrix_to_quaternion(R,order=kornia.geometry.conversions.QuaternionCoeffOrder.WXYZ).unsqueeze(0)
    rot_1 = rot_1[0].detach().cpu().numpy()
    return rot_1

from . import speedscore
def evaluate_epoch(model,data_loader, device, config,aug_intensity):
    
    errors = []
    errors_pos = []
    errors_rot = []

    k_mat_input = get_kmat_scaled(config, device) # Intrinsic matrix
    dist_coefs  = get_coefs(config, device) # Distortion coefficients
    world_kpts = loadmat("/mnt/rhome/jbp/kelvins_esa/data/speedplus/"+ "kpts.mat")
    world_kpts = world_kpts["corners"].astype(np.float32).T

    for data in tqdm(data_loader):
        data = dict_to_device(data, device)
        data = augmentate_images(data, aug_intensity)

        npts = 8
        #model.eval()
        pose_gt = data["pose"]     
        quat_gt = data["quat"]       

        t_gt= pose_gt[:,3:6]

        q_gt_b = np.expand_dims(tensor_to_mat(quat_gt[0]),axis=1)

        t_gt_b = np.expand_dims(tensor_to_mat(t_gt[0]),axis=1)

        k_mat = tensor_to_mat(k_mat_input[0]).copy()

        heatmap  = forward_pass(data["img"], model, config)
        pred_mask = heatmap
        pred_mask = F.interpolate(pred_mask, size=(data["img"].shape[2],data["img"].shape[3]), mode='bilinear', align_corners=True)
        pred_kpts = mask_to_points(pred_mask)
        pred_kpts = tensor_to_mat(pred_kpts[0])

        responses = np.max(np.max(tensor_to_mat(heatmap[0]),axis=1),axis=1)

        kpts_img = draw_keypoints(tensor_to_cvmat(data["img"][0]),pred_kpts)

        index_max = np.argsort(-responses)
        world_kpts2  = world_kpts.copy()
        world_kpts2  = world_kpts2[index_max,:]
        pred_kpts = pred_kpts[index_max,:]
        world_kpts2   = world_kpts2[:npts:]
        pred_kpts     = pred_kpts[:npts,:]        
        aux_0, rvecs_0, tvecs_0, inliers= cv2.solvePnPRansac(world_kpts2, pred_kpts, k_mat, distCoeffs=None,confidence=0.999,reprojectionError=2.0,flags=cv2.SOLVEPNP_EPNP)
        pos = tvecs_0
        rot = rvecs_0

        if True:

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

            error_0, error_pos_0, error_ori_0 = speedscore.speed_score(pos, rot, t_gt_b, q_gt_b, applyThresh=False, rotThresh=0.5, posThresh=0.005)

            errors.append(error_0)
            errors_rot.append(error_ori_0)
            errors_pos.append(error_pos_0)

    print("total samples", len(errors))
    print("Mean error: ", np.mean(errors))
    print("mean error rot: ", np.mean(errors_rot))
    print("mean error pos: ", np.mean(errors_pos))
    print("------------------------")      

    return np.mean(errors), np.mean(errors_rot), np.mean(errors_pos)


def render_heatmap(p_pnp_tgt,config, flag_visibility):
    
    h_pnp_tgt = kornia.geometry.render_gaussian2d(mean=p_pnp_tgt,
                                                 std=50*torch.ones_like(p_pnp_tgt,requires_grad=True),
                                                 size=config["target_size"],
                                                 normalized_coordinates=False)*1000/3
    
    for batch in range(h_pnp_tgt.shape[0]):
        #extract batch
        h_pnp_tgt_batch = h_pnp_tgt[batch]
        flag = flag_visibility[batch].type(torch.bool)
        
        for channel in range(h_pnp_tgt_batch.shape[0]):
            if not flag[channel]:
                h_pnp_tgt_batch[channel] = torch.zeros_like(h_pnp_tgt_batch[channel])
                
                
        
    # normalize h_pnp_tgt of size BxNxHxW with the maximum in the N dimension
    #max_val = torch.max(torch.max(h_pnp_tgt, dim=3, keepdim=False)[0],dim=2, keepdim=False)[0]
    #h_pnp_tgt = h_pnp_tgt/(max_val.unsqueeze(2).unsqueeze(3)+1e-16)
    return h_pnp_tgt


def pnp_loss(p_tgt, p_pnp_tgt, fun, config):
    
    

    p_tgt  = torch.clip(p_tgt/(config["target_size"][0]-1),0,1)
    p_pnp_tgt  = torch.clip(p_pnp_tgt/(config["target_size"][0]-1),  0, 1)

    return fun(p_pnp_tgt, p_tgt)


def pnp_loss_w_visibility(p_tgt, p_pnp_tgt, fun, config, flag):
    
    loss_pnp = 0
    for batch_index in range(p_tgt.shape[0]):
        flag_vis = flag[batch_index].type(torch.bool)
        
        p0 = p_tgt[batch_index, flag_vis, :].unsqueeze(0)
        
        p1 = p_pnp_tgt[batch_index, flag_vis, :].unsqueeze(0)
        
        
        p0  = torch.clip(p0/(config["target_size"][0]-1),0,1)
        p1  = torch.clip(p1/(config["target_size"][0]-1),  0, 1)

        loss_pnp += fun(p1, p0)
        
    loss_pnp = loss_pnp/p_tgt.shape[0]
    
 
    return loss_pnp


def compute_pair(feat_A, pose_A, pose_B, im_B, mlp_encoder, im_decoder, data, l1):

    T_AB = relative_rt(pose_A, pose_B)
    feat_pose_AB = mlp_encoder(torch.flatten(T_AB,start_dim=1)).unsqueeze(-1).unsqueeze(-1).repeat(1,1,16,16)
    
    im_pred_B = im_decoder(torch.cat([feat_A,feat_pose_AB],dim=1))
    im_pred_B = F.interpolate(im_pred_B, size=data["img"].shape[-2:], mode="bilinear", align_corners=True)

    loss_l1 = l1(im_pred_B, im_B.mean(1,True)).mean()
    

    return loss_l1



def log_results(writer, image_pred_tgt_prev, data, image_pred_tgt_next, h_prev, h_tgt, h_next, loss_recon, loss_heatmap, loss, i_iter, epoch, train_loader):
    # Make a mosaic of the images
    a = tensor_to_cvmat(image_pred_tgt_prev[0])
    b = tensor_to_cvmat(data["img"][0])
    c = tensor_to_cvmat(image_pred_tgt_next[0])
    d = np.hstack((a,b,c))
    writer.add_image("Images/PredPrev-Target-PredNext", d, i_iter + epoch * len(train_loader), dataformats="HWC")

    a = tensor_to_cvmat(data["img_prev"][0])
    b = tensor_to_cvmat(data["img"][0])
    c = tensor_to_cvmat(data["img_next"][0])
    d = np.hstack((a,b,c))
    writer.add_image("Images/Prev-Target-Next", d, i_iter + epoch * len(train_loader), dataformats="HWC")

    # Heatmap
    a = tensor_to_cvmat(h_prev[0].sum(0,True))
    b = tensor_to_cvmat(h_tgt[0].sum(0,True))
    c = tensor_to_cvmat(h_next[0].sum(0,True))
    d = np.hstack((a,b,c))
    writer.add_image("Heatmaps/PredPrev-PredTarget-PredNext", d, i_iter + epoch * len(train_loader), dataformats="HWC")

    # Heatmap
    a = tensor_to_cvmat(data["heatmap_prev"][0].sum(0,True))
    b = tensor_to_cvmat(data["heatmap"][0].sum(0,True))
    c = tensor_to_cvmat(data["heatmap_next"][0].sum(0,True))
    d = np.hstack((a,b,c))
    writer.add_image("Heatmaps/GTPrev-GTTarget-GTNext", d, i_iter + epoch * len(train_loader), dataformats="HWC")                        

    writer.add_scalar("Losses/Reconstruction", loss_recon.item(), i_iter + epoch * len(train_loader))
    writer.add_scalar("Losses/Heatmap", loss_heatmap.item(), i_iter + epoch * len(train_loader))
    writer.add_scalar("Losses/Total", loss.item(), i_iter + epoch * len(train_loader))



def log_metrics_train(writer, data, image_pred_tgt_prev, image_pred_tgt_next,
                h_prev, h_tgt, h_next, h_pnp_prev, h_pnp_tgt, h_pnp_next,  loss_recon, loss_heatmap, loss, loss_image_prev, loss_image_next,
                loss_pnp_sup, loss_pnp_unsup, loss_render=0, loss_render_sup=0, loss_unsup=0, loss_sup=0, config=None, i_iter=0, epoch=0,train_loader=None):
    
           # Make a mosaic of the images
            a = tensor_to_cvmat(image_pred_tgt_prev[0])
            b = tensor_to_cvmat(data["img"][0])
            c = tensor_to_cvmat(image_pred_tgt_next[0])
            d = np.hstack((a,b,c))
            writer.add_image("Images/PredPrev-Target-PredNext", d, i_iter + epoch * len(train_loader), dataformats="HWC")

            a = tensor_to_cvmat(data["img_prev"][0])
            b = tensor_to_cvmat(data["img"][0])
            c = tensor_to_cvmat(data["img_next"][0])
            d = np.hstack((a,b,c))
            writer.add_image("Images/Prev-Target-Next", d, i_iter + epoch * len(train_loader), dataformats="HWC")


            # Heatmap
            a = tensor_to_cvmat(h_prev[0].sum(0,True))
            b = tensor_to_cvmat(h_tgt[0].sum(0,True))
            c = tensor_to_cvmat(h_next[0].sum(0,True))
            d = np.hstack((a,b,c))
            writer.add_image("Heatmaps/PredPrev-PredTarget-PredNext", d, i_iter + epoch * len(train_loader), dataformats="HWC")


            # Heatmap
            a = tensor_to_cvmat(h_pnp_prev[0].sum(0,True))
            b = tensor_to_cvmat(h_pnp_tgt[0].sum(0,True))
            c = tensor_to_cvmat(h_pnp_next[0].sum(0,True))
            d = np.hstack((a,b,c))
            writer.add_image("Heatmaps/PnPPredPrev-PnPPredTarget-PnPPredNext", d, i_iter + epoch * len(train_loader), dataformats="HWC")



            # Heatmap
            a = tensor_to_cvmat(data["heatmap_prev"][0].sum(0,True))
            b = tensor_to_cvmat(data["heatmap"][0].sum(0,True))
            c = tensor_to_cvmat(data["heatmap_next"][0].sum(0,True))
            d = np.hstack((a,b,c))
            writer.add_image("Heatmaps/GTPrev-GTTarget-GTNext", d, i_iter + epoch * len(train_loader), dataformats="HWC")                        


            writer.add_scalar("Losses/Reconstruction", loss_recon.item(), i_iter + epoch * len(train_loader))
            writer.add_scalar("Losses/Heatmap", loss_heatmap.item(), i_iter + epoch * len(train_loader))
            writer.add_scalar("Losses/Total", loss.item(), i_iter + epoch * len(train_loader))
            writer.add_scalar("Losses/ReconstructionPrev", loss_image_prev.mean().item(), i_iter + epoch * len(train_loader))
            writer.add_scalar("Losses/ReconstructionNext", loss_image_next.mean().item(), i_iter + epoch * len(train_loader))
            writer.add_scalar("Losses/PnP Sup", loss_pnp_sup.item(), i_iter + epoch * len(train_loader))
            
            if config["activate_pnp"]:
                writer.add_scalar("Losses/PnP UnSup", loss_pnp_unsup.item(), i_iter + epoch * len(train_loader))
            if config["activate_render"]:
                writer.add_scalar("Losses/Render", loss_render.item(), i_iter + epoch * len(train_loader))
                writer.add_scalar("Losses/RenderSup", loss_render_sup.item(), i_iter + epoch * len(train_loader))

            writer.add_scalar("Unsupervised", loss_unsup.item(), i_iter + epoch * len(train_loader))
            writer.add_scalar("Supervised", loss_sup.item(), i_iter + epoch * len(train_loader))


def save_models(encoder, im_decoder, kpt_decoder, mlp_encoder, config, name_config, epoch):
    # Save at the end of the training
    # make the save dir if it doesn't exist

    path_save = os.path.join(config["save_dir"], name_config)
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    print("Saving models at epoch {}".format(epoch), "in {}".format(path_save))

    torch.save(encoder.state_dict(),        os.path.join(path_save, "encoder_model_{}.pth".format(epoch)))
    torch.save(im_decoder.state_dict(),     os.path.join(path_save, "im_decoder_model_{}.pth".format(epoch)))
    torch.save(kpt_decoder.state_dict(),    os.path.join(path_save, "kpt_decoder_model_{}.pth".format(epoch)))
    torch.save(mlp_encoder.state_dict(),    os.path.join(path_save, "mlp_encoder_model_{}.pth".format(epoch)))
    
    

def dcm2quat_2(R):
    #rot_1 = torch.from_numpy(R).unsqueeze(0).to(device)
    rot_1 = kornia.geometry.rotation_matrix_to_quaternion(R,order=kornia.geometry.conversions.QuaternionCoeffOrder.WXYZ).unsqueeze(0)
    rot_1 = rot_1[0].detach().cpu().numpy()
    return rot_1

import math

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

    # Return the Euler angles as a tuple (roll, pitch, yaw)
    return roll, pitch, yaw




def calculate_loss_heatmap(h_tgt, h_gt, flag, config, mse):
    loss_heatmap = 0.0
    for batch_index in range(h_tgt.shape[0]):
        # Get visible key-points
        flag_vis = flag[batch_index].type(torch.bool)
        # Get the batch predictions and ground-truth
        pred_batch = h_tgt[batch_index, flag_vis, :, :]
        gt_batch = config["w_heatmap"] * h_gt[batch_index, flag_vis, :, :]
        loss_heatmap += mse(pred_batch, gt_batch)
    loss_heatmap = loss_heatmap / h_tgt.shape[0] 
    return loss_heatmap


def compute_pose_w_visibility(p_tgt, kpts_world,k_mat_input,flag, pnp_fast):
    
    
    rt_tgt = []
    
    for batch_index in range(p_tgt.shape[0]):
        
        flag_vis = flag[batch_index].type(torch.bool)
        
        p_2d = p_tgt[batch_index, flag_vis, :].unsqueeze(0)
        p_3d = kpts_world[batch_index, flag_vis, :]
        k    = k_mat_input[batch_index]

        rt_tgt.append(pnp_fast(p_2d, p_3d, k))

        
    # return as a tensor
    return torch.cat(rt_tgt, dim=0)


def flag_points_inside_image(p_pnp, shape, margin):
    
    # function to flag points outside image range
    # p_pnp: 2D points in image coordinates (batch_size, n_points, 2)
    # shape: image shape (height, width)
    
    # get the image shape
    height, width = shape
    
    # get the points
    x = p_pnp[:,:,0]
    y = p_pnp[:,:,1]
    
    # flag points outside image
    
    flag_x = (x > 0 + margin) & (x < (width-margin))
    flag_y = (y > 0 + margin) & (y < (height-margin))
    
    # return the flag
    return flag_x & flag_y


def subidx(n:int, remove:int) -> list:
    out = list(range(n))
    out.remove(remove)
    return out

def submatrix(A:torch.tensor, j:int) -> torch.tensor:
    n = A.shape[0]
    s = subidx(n, j)
    o = A[s, :]
    o = o[:, s]
    return o

def get_eigenvector_val_old(hermitian_matrix, i, j, disable_complex=False):
    # Old way
    lam, v = torch.linalg.eig(hermitian_matrix)
    if disable_complex:
        old_eigenvector_ij = torch.abs(v[j,i]**2)
    else:
        old_eigenvector_ij = v[j,i]**2
    
    return old_eigenvector_ij

def get_eigenvector_val(hermitian_matrix, i, j, disable_complex=False):
    # Only need to calculate eigenvalues
    lam  = torch.linalg.eigvals(hermitian_matrix)
    
    n = len(lam)
    M = submatrix(hermitian_matrix, j)
    # Only need to calculate eigenvalues
    lam_submatrix = torch.linalg.eigvals(M)

    # Left side of equation 2
    left = torch.prod(torch.tensor([lam[i] - lam[k] for k in range(n) if k!=i]))
    # Right side of equation 2
    right = torch.prod(torch.tensor([lam[i] - lam_submatrix[k] for k in range(n-1)]))
    # Right divided by left
    eigenvector_ij = right / left
    
    if disable_complex:
        eigenvector_ij = torch.abs(eigenvector_ij)
    
    return eigenvector_ij


# Helper function to get the last available model file
def get_last_model_file(model_dir, prefix):
    model_files = [f for f in os.listdir(model_dir) if f.startswith(prefix) and f.endswith(".pth")]
    model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return model_files[-1] if model_files else None