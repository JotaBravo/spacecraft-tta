#python pretrain.py configs/pretraining/delta_5_augmentations_roe1.json
#python pretrain.py configs/pretraining/delta_5_augmentations_roe2.json

#python train.py configs/training/delta_5_augmentations_roe1.json
#python train.py configs/training/delta_5_augmentations_roe2.json

python adapt.py configs/adaptation/adaptative/delta_5_augmentations_roe1.json
python adapt.py configs/adaptation/adaptative/delta_5_augmentations_roe2.json