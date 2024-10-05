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


name_config = sys.argv[1]
config = utils.load_json(name_config)
device = config["device"]


# augmmentations
transforms = A.build_transforms(config["target_size"])


tango_loader = loader.TangoVideoLoader(config,input_transforms=transforms)
train_loader = DataLoader(tango_loader, batch_size=config["batch_size"], shuffle=True,
                          num_workers=config["num_workers"], drop_last=True)

encoder = poseresnet.get_encoder(50)
encoder.load_state_dict(torch.load("./weights/encoder_speedplus.pth"))
encoder.to(device)
encoder.train()

im_decoder = poseresnet.get_decoder(50, final_layer=1, inplanes=2048 + 1024)
im_decoder.to(device)
im_decoder.train()

kpt_decoder = poseresnet.get_decoder(50, final_layer=11)
kpt_decoder.load_state_dict(torch.load("./weights/decoder_speedplus.pth"))
kpt_decoder.to(device)
kpt_decoder.train()

mlp_encoder = poseresnet.get_mlp()
mlp_encoder.to(device)
mlp_encoder.train()

writer = tensorboard.SummaryWriter("runs/" + name_config)

# Optimize model and predictor.
optim_params = [    
    {"params": encoder.parameters(),   "lr": config["lr"]},
    {"params": im_decoder.parameters(), "lr": config["lr"]},
    {"params": kpt_decoder.parameters(), "lr": config["lr"]},
    {"params": mlp_encoder.parameters(), "lr": config["lr"]}
]

optimizer = torch.optim.Adam(optim_params)
pnp_fast = BPnP.BPnP_fast.apply

k_mat_input = utils.get_kmat_scaled(config, device)  # Intrinsic matrix
dist_coefs = utils.get_coefs(config, device)  # Distortion coefficients
kpts_world = utils.get_world_kpts(config, device)  # Spacecraft key-points

mean = config["mean_img"]
std = config["std_img"]

mse = torch.nn.MSELoss()
l1 = torch.nn.L1Loss(reduction="none")

k_mat_input = utils.get_kmat_scaled(config, device)  # Intrinsic matrix
dist_coefs = utils.get_coefs(config, device)  # Distortion coefficients
kpts_world = utils.get_world_kpts(config, device)

for epoch in range(config["epochs"]):


    mean_loss = 0
    for i_iter, data in enumerate(tqdm(train_loader)):
        
        
        data = utils.dict_to_device(data, device)
        
        p_tgt_gt  = data["kpts"]
        p_prev_gt = data["kpts_prev"]
        p_next_gt = data["kpts_next"]
        
    
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
        h_tgt  = F.interpolate(h_tgt, size=data["img"].shape[-2:], mode="bilinear",  align_corners=True)
        h_next = F.interpolate(h_next, size=data["img"].shape[-2:], mode="bilinear", align_corners=True)

        # GT pose
        rt_tgt  = data["pose"]
        rt_prev = data["pose_prev"]
        rt_next = data["pose_next"] 

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

        loss_recon   = config["w_recon"]*(loss_image_prev.mean() + loss_image_next.mean()) / 2.0
        
        # Target
        loss_heatmap = 0
        loss_heatmap += utils.calculate_loss_heatmap(h_tgt,   data["heatmap"],      data["flag_outside"],      config, mse)
        loss_heatmap += utils.calculate_loss_heatmap(h_prev,  data["heatmap_prev"], data["flag_outside_prev"], config, mse)
        loss_heatmap += utils.calculate_loss_heatmap(h_next,  data["heatmap_next"], data["flag_outside_next"], config, mse)
        loss_heatmap = loss_heatmap / 3.0

        loss = loss_recon + loss_heatmap

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mean_loss += loss.item()

        if i_iter % config["save_per_epoch"] == 0:
            utils.log_results(writer, image_pred_tgt_prev, data, image_pred_tgt_next, h_prev, h_tgt, h_next, loss_recon, loss_heatmap, loss, i_iter, epoch, train_loader)
       
    mean_loss /= len(train_loader)
    print("Epoch: %d, Loss: %f" % (epoch, mean_loss))
    writer.add_scalar("Losses/Epoch", mean_loss, epoch)

    if not epoch%50:
        utils.save_models(encoder, im_decoder, kpt_decoder, mlp_encoder, config, name_config, epoch)

utils.save_models(encoder, im_decoder, kpt_decoder, mlp_encoder, config, name_config, epoch)
