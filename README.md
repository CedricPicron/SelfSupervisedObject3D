# SelfSupervisedObject3D
Method for self-supervised monocular 3D object detection, demonstrated for 3D car detection in the autonomous driving setting.

<p align="center"> <img src='source/Kitti/Tracking/Results/Video3D/4.gif' align="center" width="100%"> </p> 

The method consists of two steps:
1. A network is trained in a self-supervised way to detect the yaw angle orientations of the different cars in the scene.
2. An optimization method is used to maximize the 2D IoU of the estimated 2D box with the projection of the estimated 3D box.

## Installation
We provide instructions how to install dependencies via conda. First, clone the repository locally:
```
git clone https://github.com/CedricPicron/SelfSupervisedObject3D
```
Then, install PyTorch 1.5+ and torchvision 0.6+:
```
conda install -c pytorch pytorch torchvision
```
Finally, make sure that `numpy`, `pandas` and `scipy` are also present in the conda environment.

## Data preparation
We make use of three datasets: Kitti, nuScenes and Virtual Kitti. Below we specify which directories need to be completed. Here *completion* can be achieved *directly* by placing the correct data into the directories, or *indirectly* by creating symlinks to the real data location.

#### Kitti
For Kitti, we use both the *3D object detection* and the *tracking* datasets. 
* For the **3D object detection** dataset, complete the `calib`, `image_2` and `label_2` directories under `datasets/Kitti/Object3D/training`. 
* For the **tracking** dataset, complete the `calib`, `image_02` and `label_02` directories under `datasets/Kitti/Tracking/training`.

#### nuScenes
For nuScenes, we support both `v1.0-mini` and `v1.0-trainval` versions natively. For each version, complete both the `datasets/NuScenes/<version>/samples` and `datasets/NuScenes/<version>/<version>` directories.

#### Virtual Kitti
For Virtual Kitti, simply complete the `datasets/VirtualKitti` directory, as obtained by downloading the dataset. We assume the 1.3.1 version of the dataset.

## Usage
For usage, simply run the desired python scripts found under `source/<dataset>/Scripts`. Beware, some scripts require trained models. Therefore, the scripts are best run in following order:

1. First, run the `angleEstimator.py` scripts to obtain pretrained angle estimators (available Kitti tracking, nuScenes and Virtual Kitti).
2. Secondly, load the pretrained model and fine-tune it in a self-supervised way with `selfSupervisedAngle.py` (available for Kitti tracking and nuScenes). Default fine-tunes model pretrained on Virtual Kitti.
3. Thirdly, perform the 3D optimization method with the angle model from previous step in `optimization3D.py` and evaluate using Kitti metric (available for Kitti 3D object detection and nuScenes).
4. Finally, obtain some videos of 3D car detections with `video3D.py` (available for Kitti tracking and nuScenes).

In steps 2-4, make sure the correct models are loaded in (see command-line arguments of corresponding script for more information). Finally, note that some of our obtained results are found under `source/<dataset>/Results/<experiment>`.
