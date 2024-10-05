import os
import sys

import kornia.augmentation as K
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import tensorboard
from torch.utils.data import DataLoader
from tqdm import tqdm

from BPnP import BPnP
from loaders import loader
from models import poseresnet
from src import augmentations as A
from src import utils

# Set the seed for reproducibility.
seed = 1984
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

config_file = sys.argv[1]
config = utils.load_json(config_file)
device = config["device"]

transforms = A.build_transforms(config["target_size"])


tango_loader = loader.TangoVideoLoader(config,input_transforms=transforms)
train_loader = DataLoader(tango_loader, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"],drop_last=True)


# Define the model directory based on config

name_config = config_file.split("/")[-1]
model_dir = os.path.join(config["path_pretrained"], name_config)


# Load the last encoder model
encoder = poseresnet.get_encoder(50)
encoder.to(device)
last_encoder_model = utils.get_last_model_file(model_dir, "encoder_model_")
if last_encoder_model:
    encoder.load_state_dict(torch.load(os.path.join(model_dir,last_encoder_model)))
encoder.train()

# Load the last kpt_decoder model
kpt_decoder = poseresnet.get_decoder(50, final_layer=11)
kpt_decoder.to(device)
last_kpt_decoder_model = utils.get_last_model_file(model_dir, "kpt_decoder_model_")
if last_kpt_decoder_model:
    kpt_decoder.load_state_dict(torch.load(os.path.join(model_dir,last_kpt_decoder_model)))
kpt_decoder.train()

# Load the last im_decoder model (image decoder)
im_decoder = poseresnet.get_decoder(50, final_layer=1, inplanes=2048 + 1024)
im_decoder.to(device)
last_im_decoder_model = utils.get_last_model_file(model_dir,"im_decoder_model_")
if last_im_decoder_model:
    im_decoder.load_state_dict(torch.load(os.path.join(model_dir, last_im_decoder_model)))
im_decoder.train()

# Load the last mlp_encoder model
mlp_encoder = poseresnet.get_mlp()
mlp_encoder.to(device)
last_mlp_encoder_model = utils.get_last_model_file(model_dir,"mlp_encoder_model_")
if last_mlp_encoder_model:
    mlp_encoder.load_state_dict(torch.load(os.path.join(model_dir, last_mlp_encoder_model)))
mlp_encoder.train()


writer = tensorboard.SummaryWriter("runs/" + name_config)

# optimize model and predictor

optim_params = [{"params": encoder.parameters(), "lr": config["lr"]},
                    {"params": im_decoder.parameters(), "lr": config["lr"]},
                    {"params": mlp_encoder.parameters(), "lr": config["lr"]},              
                    {"params": kpt_decoder.parameters(), "lr": config["lr"]},
    ]


optimizer = torch.optim.Adam(optim_params)


pnp_fast   = BPnP.BPnP_fast.apply

k_mat_input = utils.get_kmat_scaled(config, device) # Intrinsic matrix
dist_coefs  = utils.get_coefs(config, device) # Distortion coefficients
kpts_world  = utils.get_world_kpts(config,device) # Spacecraft key-points

mean        = config["mean_img"]
std         = config["std_img"]


mse  = torch.nn.MSELoss()
l1   = torch.nn.L1Loss(reduction="none")    

k_mat_input = utils.get_kmat_scaled(config, device) # Intrinsic matrix
dist_coefs  = utils.get_coefs(config, device) # Distortion coefficients
kpts_world  = utils.get_world_kpts(config, device) 


for epoch in range(config["epochs"]):

    mean_loss = 0
    for i_iter, data in enumerate(tqdm(train_loader)):
        
        data = utils.dict_to_device(data, device)
       
       
        
        p_tgt_gt  = utils.heatmap_to_points(data["heatmap"], temp=config["temp"])
        p_prev_gt = utils.heatmap_to_points(data["heatmap_prev"], temp=config["temp"])
        p_next_gt = utils.heatmap_to_points(data["heatmap_next"], temp=config["temp"])

        # Image embeddings
        feat_prev = encoder(data["img_prev"])
        feat_tgt  = encoder(data["img"])
        feat_next = encoder(data["img_next"])

        # Decode heatmaps
        h_prev = kpt_decoder(feat_prev)
        h_tgt  = kpt_decoder(feat_tgt)
        h_next = kpt_decoder(feat_next)

        # Scale up things for the loss computation
        h_prev = F.interpolate(h_prev, size=data["img"].shape[-2:], mode="bilinear", align_corners=True)
        h_tgt  = F.interpolate(h_tgt, size=data["img"].shape[-2:], mode="bilinear", align_corners=True)
        h_next = F.interpolate(h_next, size=data["img"].shape[-2:], mode="bilinear", align_corners=True)

        # Pose embedding
        p_prev   = utils.heatmap_to_points(h_prev, temp=config["temp"])
        p_tgt    = utils.heatmap_to_points(h_tgt, temp=config["temp"])
        p_next   = utils.heatmap_to_points(h_next, temp=config["temp"])


        rt_tgt = utils.compute_pose_w_visibility(p_tgt, kpts_world,k_mat_input,data["flag_outside"], pnp_fast)
            
        p_pnp_tgt = BPnP.batch_project(rt_tgt, kpts_world[0], k_mat_input[0])
        flag_tgt = utils.flag_points_inside_image(p_pnp_tgt, data["img"].shape[-2:],config["safe_margin_kpts"])
        data["flag_outside"] = torch.bitwise_and(data["flag_outside"], flag_tgt)

        rt_prev = utils.compute_pose_w_visibility(p_prev, kpts_world,k_mat_input,data["flag_outside_prev"], pnp_fast)
        p_pnp_prev = BPnP.batch_project(rt_prev, kpts_world[0], k_mat_input[0])

        # flag keypoints falling outside image range
        flag_prev = utils.flag_points_inside_image(p_pnp_prev, data["img"].shape[-2:],config["safe_margin_kpts"])
        data["flag_outside_prev"] = torch.bitwise_and(data["flag_outside_prev"], flag_prev)


        rt_next = utils.compute_pose_w_visibility(p_next, kpts_world,k_mat_input,data["flag_outside_next"], pnp_fast)          
        p_pnp_next = BPnP.batch_project(rt_next, kpts_world[0], k_mat_input[0])
        flag_next = utils.flag_points_inside_image(p_pnp_next, data["img"].shape[-2:],config["safe_margin_kpts"])
        data["flag_outside_next"] = torch.bitwise_and(data["flag_outside_next"], flag_next)

        T_tgt_prev = utils.relative_rt(rt_prev, rt_tgt)

        T_tgt_next = utils.relative_rt(rt_next, rt_tgt)

        feat_pose_tgt_prev = mlp_encoder(torch.flatten(T_tgt_prev,start_dim=1)).unsqueeze(-1).unsqueeze(-1).repeat(1,1,16,16)
        feat_pose_tgt_next = mlp_encoder(torch.flatten(T_tgt_next,start_dim=1)).unsqueeze(-1).unsqueeze(-1).repeat(1,1,16,16)

                
        # Decode images
        image_pred_tgt_prev = im_decoder(torch.cat([feat_prev,feat_pose_tgt_prev],dim=1))
        image_pred_tgt_next = im_decoder(torch.cat([feat_next,feat_pose_tgt_next],dim=1))

        image_pred_tgt_prev = F.interpolate(image_pred_tgt_prev, size=data["img"].shape[-2:], mode="bilinear", align_corners=True)
        image_pred_tgt_next = F.interpolate(image_pred_tgt_next, size=data["img"].shape[-2:], mode="bilinear", align_corners=True)



        loss_image_prev = l1(image_pred_tgt_prev, data["img"].mean(1,True))
        loss_image_next = l1(image_pred_tgt_next, data["img"].mean(1,True))
        loss_recon = config["w_recon"]*(loss_image_prev.mean() + loss_image_next.mean()) / 2.0

        loss_unsup = loss_recon

        # Target
        loss_heatmap = 0
        loss_heatmap += utils.calculate_loss_heatmap(h_tgt,  data["heatmap"],      data["flag_outside"],       config, mse)
        loss_heatmap += utils.calculate_loss_heatmap(h_prev, data["heatmap_prev"], data["flag_outside_prev"],  config, mse)
        loss_heatmap += utils.calculate_loss_heatmap(h_next, data["heatmap_next"], data["flag_outside_next"],  config, mse)
        loss_heatmap = loss_heatmap / 3.0
        loss_sup = loss_heatmap
            
        # Losses regularization
        if config["activate_pnp"]:

            loss_pnp_unsup = config["w_pnp"]*(utils.pnp_loss_w_visibility(p_tgt, p_pnp_tgt, mse, config,data["flag_outside"]) + utils.pnp_loss_w_visibility(p_next, p_pnp_next, mse, config,data["flag_outside_next"]) + utils.pnp_loss_w_visibility(p_prev, p_pnp_prev, mse, config,data["flag_outside_prev"]))/3.0
            loss_unsup+=loss_pnp_unsup
                
            loss_pnp_sup = config["w_pnp"]*(utils.pnp_loss_w_visibility(p_pnp_tgt, p_tgt_gt, mse, config,data["flag_outside"]) + utils.pnp_loss_w_visibility(p_pnp_next, p_next_gt, mse, config,data["flag_outside_next"]) + utils.pnp_loss_w_visibility(p_pnp_prev, p_prev_gt, mse, config,data["flag_outside_prev"]))/3.0
            loss_sup += loss_pnp_sup

  
        if config["activate_render"]:
            h_pnp_prev  = utils.render_heatmap(p_pnp_prev, config,data["flag_outside"])
            h_pnp_tgt   = utils.render_heatmap(p_pnp_tgt, config,data["flag_outside_prev"])
            h_pnp_next  = utils.render_heatmap(p_pnp_next, config,data["flag_outside_next"])

            loss_render = 0
            loss_render += utils.calculate_loss_heatmap(h_tgt,  h_pnp_tgt,  data["flag_outside"],       config, mse)
            loss_render += utils.calculate_loss_heatmap(h_prev, h_pnp_prev, data["flag_outside_prev"],  config, mse)
            loss_render += utils.calculate_loss_heatmap(h_next, h_pnp_next, data["flag_outside_next"],  config, mse)            
            loss_render = config["w_render"]*(loss_render / 3.0)

            loss_unsup+=loss_render

            loss_unsup/=(int(config["activate_pnp"]) + int(config["activate_render"]) + 1)

            loss_sup /= (int(config["activate_pnp"]) + int(config["activate_render"]) + 1)


        
        loss = loss_unsup + loss_sup

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mean_loss += loss.item()



        if i_iter % config["save_per_epoch"] == 0:

            if not config["activate_render"]:
                loss_render = torch.tensor(0)
                h_pnp_prev = torch.zeros_like(h_prev)
                h_pnp_tgt = torch.zeros_like(h_tgt)
                h_pnp_next = torch.zeros_like(h_next)
            if not config["activate_pnp"]:
                loss_pnp_unsup = torch.tensor(0)
                loss_pnp_sup = torch.tensor(0)
   
            utils.log_metrics_train(writer, data, image_pred_tgt_prev, image_pred_tgt_next,
                                        h_prev, h_tgt, h_next, h_pnp_prev, h_pnp_tgt, h_pnp_next, loss_recon, loss_heatmap, loss,
                                        loss_image_prev, loss_image_next, loss_pnp_sup, loss_pnp_unsup, loss_render, torch.tensor(0), loss_unsup, loss_sup, config, i_iter, epoch,
                                        train_loader)
                


    mean_loss /= len(train_loader)
    print("Epoch: %d, Loss: %f" % (epoch, mean_loss))
    writer.add_scalar("Losses/Epoch", mean_loss, epoch)


utils.save_models(encoder, im_decoder, kpt_decoder, mlp_encoder, config, config_file, epoch)
