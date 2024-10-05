# Test-Time Adaptation for Keypoint-Based Spacecraft Pose Estimation Based on Predicted-View Synthesis

Official Pytorch Implementation of Test-Time Adaptation for Keypoint-Based Spacecraft Pose Estimation Based on Predicted-View Synthesis by Juan Ignacio Bravo Pérez-Villar, Álvaro García-Martín, Jesús Bescós, and Juan C. San-Miguel  (IEEE Transactions on Aerospace and Electronic Systems).

![novel-view-VII-final-1](https://github.com/user-attachments/assets/6e77d2ac-aab9-4dbb-b1e9-722aa8ebcbee)


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

## Summary
This work proposes a test-time adaptation approach that leverages the temporal redundancy between images acquired during close proximity operations. Our approach involves extracting features from sequential spacecraft images, estimating their poses, and then using this information to synthesise a reconstructed view. We establish a self-supervised learning objective by comparing the synthesised view with the actual one. During training, we supervise both pose estimation and image synthesis, while at test-time, we optimise the self-supervised objective. Additionally, we introduce a regularisation loss to prevent solutions that are not consistent with the keypoint structure of the spacecraft. 

## Setup

This section contains the instructions to execute the code. The repository has been tested in a system with:
- Ubuntu 18.04
- CUDA 11.2
- Conda 4.8.3

### Clone the repository

### 2.2. Clone Repository and create a Conda environment
To clone the repository, type in your terminal:

```
git clone https://github.com/JotaBravo/spacecraft-tta.git
```

After [instaling conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) go to the spacecraft-tta folder and type in your terminal:

``` 
conda env create -f env.yml
conda activate spacecraft-tta
```

### Download the datasets and generate the heatmaps

This work employs the [SHIRT Dataset](https://purl.stanford.edu/zq716br5462) dataset.  
SHIRT provides the ground-truth information as pairs of images and poses (relative position and orientation of the spacecraft w.r.t the camera). Our method assumes the ground-truth is provided as key-point maps. 


Place the contents of the dataset under the dataset folder. The folder should look:
```
dataset
    roe 1
        |- synthetic

roe 2
    | - synthetic

camera.json
license.md
METADATA.md
``` 


