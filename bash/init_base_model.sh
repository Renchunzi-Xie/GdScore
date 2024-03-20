#!/usr/bin/env bash

gpu=$1

for ARCH in resnet18 resnet50 wrn_50_2
do
    python init_base_model.py --arch ${ARCH} --train_epoch 20 --train_data_name cifar10 --lr 0.001 --batch_size 128 --seed 123 --gpu ${gpu} --cifar_data_path /data/czxie/datasets/Cifar10 --cifar_corruption_path /data/czxie/datasets/CIFAR-10-C
    python init_base_model.py --arch ${ARCH} --train_epoch 50 --train_data_name cifar100 --lr 0.001 --batch_size 128 --seed 123 --gpu ${gpu}  --cifar_data_path /data/czxie/datasets/Cifar100 --cifar_corruption_path data/czxie/datasets/CIFAR-100-C
    python init_base_model.py --arch ${ARCH} --train_epoch 50 --train_data_name tinyimagenet --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path /data/czxie/datasets/tiny-imagenet-200 --cifar_corruption_path /data/czxie/datasets/Tiny-ImageNet-C
done


