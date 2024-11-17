# GdScore
## Datasets
### Pre-trained process

1. CIFAR10 & CIFAR-100 can be downloaded in the code. 
2. Download TinyImageNet
```angular2html
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
```
3. Download ImageNet
```angular2html
https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data
```
4. Download Office-Home
```angular2html
pip install gdown==4.6.0
gdown https://drive.google.com/uc?id=1JMFEHM46xmgp2RSX6iVgcR5fCpZkeruJ
```
5. Download PACS
```angular2html
https://www.kaggle.com/datasets/nickfratto/pacs-dataset
```
6. Download DomainNet
```angular2html
wget http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip
```
### Evaluation process
1. Download CIFAR-10C & CIFAR-100C
```angular2html
curl -O https://zenodo.org/record/2535967/files/CIFAR-10-C.tar
curl -O https://zenodo.org/record/3555552/files/CIFAR-100-C.tar
```
2. Download TinyImageNet-C
```angular2html
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
```
3. Download ImageNet-C
```angular2html
curl -O https://zenodo.org/record/2235448/files/blur.tar
curl -O https://zenodo.org/record/2235448/files/digital.tar
curl -O https://zenodo.org/record/2235448/files/extra.tar
curl -O https://zenodo.org/record/2235448/files/noise.tar
curl -O https://zenodo.org/record/2235448/files/weather.tar
```
## Pre-training and evaluation

Step 1: Pre-train models using commands in `./bash/init_base_model.sh`.

Step 2: Estimate OOD error using commands in `./bash/gdscore.sh`.
