# SoftTissueTumorSegmentation

This repository contains all code to automatically segment 3D MRI scans containing sarcomas located in patients extremities using a 3D Residual UNet (3DResUNet) architecture. 

## Preliminaries

1. Setup and activate conda environment

```console
$ conda env create -f environment.yaml
```

```console
$ conda activate SoftTissueTumorSegmentation
```

2. Add MRI images and their respective segmentation masks to the [data directory](./data/). 

3. Add correct path for train/val and test files to the [config.ini](./config.ini) file. 


## Training

4. Start training, validation and testing by invoking 

```console
$ python main.py
```

## During/After Training 

5. Hyperparameters contained in the [config.ini](./config.ini) will be stored using [MLFlow](www.mlflow.org).

6. Training, validation and testing progress/results can be evaluated using the MLFlow UI

```console
$ mlflow ui
```

7. A quantitative evaluation of the segmentation results can be made using Tensorboard
```console
$ tensorboard --logdir=runs
```
