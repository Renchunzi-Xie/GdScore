#!/usr/bin/env bash

gpu=$1
threshold=$2

for ARCH in resnet18 resnet50 wrn_50_2
do
    python main.py --alg gdscore --arch ${ARCH} --dataname cifar10 --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path /data/czxie/datasets/Cifar10 --cifar_corruption_path /data/czxie/datasets/CIFAR-10-C --threshold 0.5 --norm_type 0.3
    python main.py --alg gdscore --arch ${ARCH} --dataname cifar100 --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path /data/czxie/datasets/Cifar100 --cifar_corruption_path /data/czxie/datasets/CIFAR-100-C --threshold 0.5 --norm_type 0.3
    python main.py --alg gdscore --arch ${ARCH} --dataname tinyimagenet --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path /data/czxie/datasets/tiny-imagenet-200 --cifar_corruption_path /data/czxie/datasets/Tiny-ImageNet-C --threshold 0.5 --norm_type 0.3
#    python main.py --alg gdscore --arch ${ARCH} --dataname imagenet --lr 0.001 --num_classes 1000 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path /data/DS1618p/Image__ILSVRC2012 --cifar_corruption_path /data/home/czxie/imagenet-c --threshold ${threshold} --norm_type -3
#    python main.py --alg gdscore --arch ${ARCH} --dataname pacs --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path /data/home/czxie/datasets/PACS/PACS --cifar_corruption_path /data/home/czxie/datasets/PACS/PACS --threshold ${threshold} --norm_type -3
#    python main.py --alg gdscore --arch ${ARCH} --dataname office31 --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path /data/home/czxie/datasets/office31 --cifar_corruption_path /data/home/czxie/datasets/office31 --threshold ${threshold} --norm_type -3
#    python main.py --alg gdscore --arch ${ARCH} --dataname office_home --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path /data/home/czxie/datasets/OfficeHomeDataset_10072016 --cifar_corruption_path /data/home/czxie/datasets/OfficeHomeDataset_10072016 --threshold ${threshold} --norm_type -3
#    python main.py --alg gdscore --arch ${ARCH} --dataname wilds_camelyon17 --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path /data/home/czxie --cifar_corruption_path /data/home/czxie --score external --threshold ${threshold} --norm_type -3
#    python main.py --alg gdscore --arch ${ARCH} --dataname entity13 --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path /data/DS1618p/Image__ILSVRC2012 --cifar_corruption_path /data/home/czxie/imagenet-c --threshold ${threshold} --norm_type -3
#    python main.py --alg gdscore --arch ${ARCH} --dataname entity30 --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path /data/DS1618p/Image__ILSVRC2012 --cifar_corruption_path /data/home/czxie/imagenet-c --threshold ${threshold} --norm_type -3
#    python main.py --alg gdscore --arch ${ARCH} --dataname living17 --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path /data/DS1618p/Image__ILSVRC2012 --cifar_corruption_path /data/home/czxie/imagenet-c --threshold ${threshold} --norm_type -3
#    python main.py --alg gdscore --arch ${ARCH} --dataname nonliving26 --lr 0.001 --batch_size 128 --seed 1 --score external --gpu ${gpu} --cifar_data_path /data/DS1618p/Image__ILSVRC2012 --cifar_corruption_path /data/home/czxie/imagenet-c --threshold ${threshold} --norm_type -3
done

