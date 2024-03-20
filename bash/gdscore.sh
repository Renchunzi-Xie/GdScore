#!/usr/bin/env bash

gpu=$1
threshold=$2

for ARCH in resnet18 resnet50 wrn_50_2
do
    python main.py --alg gdscore --arch ${ARCH} --dataname cifar10 --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path /data/czxie/datasets/Cifar10 --cifar_corruption_path /data/czxie/datasets/CIFAR-10-C --threshold 0.5 --norm_type 0.3
    python main.py --alg gdscore --arch ${ARCH} --dataname cifar100 --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path /data/czxie/datasets/Cifar100 --cifar_corruption_path /data/czxie/datasets/CIFAR-100-C --threshold 0.5 --norm_type 0.3
    python main.py --alg gdscore --arch ${ARCH} --dataname tinyimagenet --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path /data/czxie/datasets/tiny-imagenet-200 --cifar_corruption_path /data/czxie/datasets/Tiny-ImageNet-C --threshold 0.5 --norm_type 0.3
done

