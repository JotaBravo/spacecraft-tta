# Test-Time Adaptation for Keypoint-Based Spacecraft Pose Estimation Based on Predicted-View Synthesis

<<<<<<< HEAD
Official **PyTorch** Implementation of **Test-Time Adaptation for Keypoint-Based Spacecraft Pose Estimation Based on Predicted-View Synthesis** by Juan Ignacio Bravo Pérez-Villar, Álvaro García-Martín, Jesús Bescós, and Juan C. San-Miguel  (IEEE Transactions on Aerospace and Electronic Systems).
=======
Official Pytorch Implementation of Test-Time Adaptation for Keypoint-Based Spacecraft Pose Estimation Based on Predicted-View Synthesis by Juan Ignacio Bravo Pérez-Villar, Álvaro García-Martín, Jesús Bescós, and Juan C. San-Miguel  (IEEE Transactions on Aerospace and Electronic Systems).

![novel-view-VII-final-1](https://github.com/user-attachments/assets/6e77d2ac-aab9-4dbb-b1e9-722aa8ebcbee)
>>>>>>> 2a1f09d85b1b68ad92402f2199345551cfd1add7

![novel-view-VII-final-1](https://github.com/user-attachments/assets/6e77d2ac-aab9-4dbb-b1e9-722aa8ebcbee)



<p align="center">
    <img src="https://user-images.githubusercontent.com/22771127/185179617-e77acf05-2f93-45dc-9d2d-a9d771e48d0b.png" alt="Deimos Space Logo" style="width:15%"/>
    <img src="https://user-images.githubusercontent.com/22771127/185183738-692554f7-548b-4192-a50f-9dd2af2d4b9d.png"  alt="VPU Lab Logo" style="width:15%"/>
    <img src="https://user-images.githubusercontent.com/22771127/189942036-58e17f72-a385-4955-be07-f347e109eaba.png"  alt="VPU Lab Logo" style="width:15%"/>
</p>


## Summary
This work proposes a test-time adaptation approach that leverages the temporal redundancy between images acquired during close proximity operations. Our approach involves extracting features from sequential spacecraft images, estimating their poses, and then using this information to synthesise a reconstructed view. We establish a self-supervised learning objective by comparing the synthesised view with the actual one. During training, we supervise both pose estimation and image synthesis, while at test-time, we optimise the self-supervised objective. Additionally, we introduce a regularisation loss to prevent solutions that are not consistent with the keypoint structure of the spacecraft. 

## Setup

This section contains the instructions to execute the code. The repository has been tested in a system with:
- Ubuntu 18.04
- CUDA 11.2
- Conda 4.8.3

#### Clone Repository and create a Conda environment
To clone the repository, type in your terminal:

```
git clone https://github.com/JotaBravo/spacecraft-tta.git
```

After [instaling conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) go to the spacecraft-tta folder and type in your terminal:

``` 
conda env create -f env.yml
conda activate spacecraft-tta
```

#### Download the dataset

This work employs the [SHIRT Dataset](https://purl.stanford.edu/zq716br5462) dataset. Place the contents of the dataset under the dataset folder. The folder should look like this:
```

dataset
  roe1
    synthetic
    lightbox
  roe2
    synthetic
    lightbox
camera.json
``` 
#### Get the keypoints

Download our [keypoints](https://drive.google.com/file/d/1-dxstIkkA7MC76s9DRMMjvqCm-FucdY9/view?usp=drive_link) and copy them under the roe1 and roe2 folders.

```
dataset/roe1/kpts.mat
dataset/roe2/kpts.mat
``` 
**NOTE**: to reproduce our work the keypoints provided by us are needed. However, we encourage using the keypoints from the SPEED+ team. Those can be found in: https://github.com/tpark94/speedplusbaseline

#### Generate the heatmaps

You can download our heatmaps from our [Google Drive](https://drive.google.com/drive/folders/1Eq1ZXh78J1tFQUkAsTX3Wkxn4qFx-ACJ?usp=drive_link). Extract them and place them under their corresponding folders
```
dataset/roe1/synthetic/kptsmap
dataset/roe2/synthetic/kptsmap
```
Or simply run
```
python generate_heatmaps.py
```

## Run

You can download the checkpoints from this [Google Drive](https://drive.google.com/drive/folders/1FZgy8EjC6ghT4Ja1pqbb919ESuPEVHgi?usp=sharing) folder

#### Evaluate

To run the experiments, execute:

 ```
 python test.py <config_path> <encoder_path> <decoder_path>
 ```

 #### Train from scratch

To train from scratch you can execute. Optinally (recommended) you can download our [speedplus pretrained weights](https://drive.google.com/drive/folders/1Vfvf4DqvRFZhHjjcn4tKLd9PC1-alWhg?usp=drive_link) for the pretraining stage and place them under the weights folder. Otherwise you can comment the corresponding lines in pretrain.py

 ```
source scripts/launch_training.sh
 ```




## Cite

If you find our work or code useful, please cite:
```
@article{perez2024test,
  title={Test-Time Adaptation for Keypoint-Based Spacecraft Pose Estimation Based on Predicted-View Synthesis},
  author={P{\'e}rez-Villar, Juan Ignacio Bravo and Garc{\'\i}a-Mart{\'\i}n, {\'A}lvaro and Besc{\'o}s, Jes{\'u}s and SanMiguel, Juan C},
  journal={IEEE Transactions on Aerospace and Electronic Systems},
  year={2024},
  publisher={IEEE}
}
```


## Acknowledgments
This work is supported by Comunidad Autónoma de Madrid (Spain) under the Grant IND2020/TIC-17515
