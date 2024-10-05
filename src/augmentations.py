'''
Copyright (c) 2022 SLAB Group
Licensed under MIT License (see LICENSE.md)
Author: Tae Ha Park (tpark94@stanford.edu)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from .  import randomsunflare
from .  import coarsedropout

def build_transforms(cfg, is_train=True, load_labels=True):
    transforms = []

    # Resize the image
    transforms = [A.Resize(512,512)]

    # Add augmentation if training, skip if not
    if is_train:
        transforms += [A.RandomBrightnessContrast(brightness_limit=0.2,
                                                      contrast_limit=0.2,
                                                      p=0.5)]

        transforms += [A.CoarseDropout(max_holes=5,
                                         min_holes=1,
                                         p=0.5)]

        transforms += [A.RandomSunFlare(num_flare_circles_lower=1,
                                          num_flare_circles_upper=10,
                                          p=0.5,src_radius=33)]

        transforms += [A.OneOf(
                [
                    A.MotionBlur(blur_limit=(3,9)),
                    A.MedianBlur(blur_limit=(3,7)),
                    A.GlassBlur(sigma=0.1,
                                max_delta=2)
                ], p=0.5
            )]

        transforms += [A.OneOf(
                [
                    A.GaussNoise(var_limit=40), # variance [pix]
                    A.ISONoise(color_shift=(0.1, 0.5),
                               intensity=(0.3, 0.8))
                ], p=0.5
            )]

    # Normalize by ImageNet stats, then turn into tensor
    transforms += [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                   ToTensorV2()]

    # Compose and return
    if False:
        transforms = A.Compose(
            transforms,
            A.BboxParams(format='albumentations',       # [xmin, ymin, xmax, ymax] (normalized)
                         label_fields=['class_labels']) # Placeholder
        )
    else:
        transforms = A.Compose(
            transforms,    additional_targets={'image0': 'image', 'image1': 'image'}

        )

    return transforms

if __name__=='__main__':

    from PIL import Image
    data = Image.open('/Users/taehapark/SLAB/Dataset/speedplus/synthetic/images/img000100.jpg').convert('RGB')

    transforms = build_transforms((480, 320))

    data = transforms(data)
    data = T.functional.to_pil_image(data)

    data.show()


