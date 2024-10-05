import utils
import kornia

import torch.nn.functional as F
    
    
def augmentate_images(data, augmentation):
    for key in data:
        if "img" in key:
            data[key] = augmentation(data[key])
      
    
    
    
    
def relative_poses(data, model, pnp_fast, kpts_world, k_mat_input, config):

    # -------------------------------------------------------------------------------------- #
    # ----------------------------------------- Target ------------------------------------- #
    # -------------------------------------------------------------------------------------- #
    heatmap = model(img)[0]
    heatmap = heatmap["hm_keypoint"]
    heatmap = F.interpolate(heatmap, size=config["target_size"], mode='bilinear', align_corners=True)
        
    # Derive pose via PnP
    kpts_pred = utils.heatmap_to_points(heatmap, temp=10.0)
    rt_target = pnp_fast(kpts_pred, kpts_world[0], k_mat_input[0]) # Bx6 [rot,pose]
    R_tgt = kornia.geometry.conversions.angle_axis_to_rotation_matrix(rt_target[:,0:3])
    t_tgt = rt_target[:,3:6]
    
    