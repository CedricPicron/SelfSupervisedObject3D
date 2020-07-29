import sys
sys.path.insert(1, '../../../../datasets/Kitti/Tracking/utils/Scripts')

from utils import computeBox3D
from utils import drawBox3D
from utils import getDefaultLabel
from utils import getRayAngle
from utils import readCalibrationFile

import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import subprocess
import torch

from PIL import Image
from scipy.optimize import minimize
from torch import nn
from torchvision import models
from torchvision import transforms

''' ===================================================================== '''
''' ------------------------- Parsing arguments ------------------------- '''
''' ===================================================================== '''

argparser = argparse.ArgumentParser(description=__doc__)
argparser.add_argument(
    '--angleModelPath',
    metavar='PATH',
    type=str,
    default='../Models/SelfSupervisedAngle/All/1.pt',
    help='Path to self-supervised orientation model.')
argparser.add_argument(
    '--boxHeight',
    metavar='HEIGHT',
    type=float,
    default=1.5,
    help='Height of 3D boxes.')
argparser.add_argument(
    '--boxLength',
    metavar='LENGTH',
    type=float,
    default=3.88,
    help='Length of 3D boxes.')
argparser.add_argument(
    '--boxWidth',
    metavar='WIDTH',
    type=float,
    default=1.63,
    help='Width of 3D boxes.')
argparser.add_argument(
    '--dataRoot',
    metavar='PATH',
    type=str,
    default='../../../../datasets/Kitti/Tracking/training',
    help='Path to dataset location.')
argparser.add_argument(
    '--experimentRoot',
    metavar='PATH',
    type=str,
    default='..',
    help='Path to experiment base directory.')
argparser.add_argument(
    '--imageScaleFactor',
    metavar='FACTOR',
    type=float,
    default=0.03,
    help='Factor scaling the size of the video images.')
argparser.add_argument(
    '--minimumScore',
    metavar='SCORE',
    type=float,
    default=0.7,
    help='Minimum score needed for a detection to be shown.')
argparser.add_argument(
    '--numberOfSamples',
    metavar='SAMPLES',
    type=int,
    default=1,
    help='Number of sequences to sample.')
args = argparser.parse_args()

''' ================================================================== '''
''' ------------------------- Initialization ------------------------- '''
''' ================================================================== '''

calibDir = os.path.join(args.dataRoot, 'calib')
imageDir = os.path.join(args.dataRoot, 'image_02')

videosDir = os.path.join(args.experimentRoot, 'Results/Video3D')
os.makedirs(videosDir, exist_ok=True)

device = torch.device('cuda')
maskRCNN = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
maskRCNN.eval().to(device)
detectionTransform = transforms.Compose([transforms.ToTensor()])

angleModel = models.resnext50_32x4d()
angleModel.fc = nn.Linear(angleModel.fc.in_features, 1)
angleModel.load_state_dict(torch.load(args.angleModelPath)['model'])
angleModel.eval().to(device)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
angleTransform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

''' =================================================================== '''
''' ------------------------- 3D optimization ------------------------- '''
''' =================================================================== '''

def boxFitOptimization(calib3x4, carBox, carLabel, initialDepth=30.0):
    xCenter = 0.5 * (carLabel['left'] + carLabel['right'])
    yCenter = 0.5 * (carLabel['top'] + carLabel['bottom'])
    center2D = np.array([xCenter, yCenter, 1])

    invCalib3x3 = np.linalg.inv(calib3x4[:, :-1])
    center3D = invCalib3x3.dot(center2D)
    normCenter3D = center3D[:-1]/center3D[2]

    functionArgs = (calib3x4, carBox, carLabel, normCenter3D)
    res = minimize(functionToMinimize, x0=initialDepth, args=functionArgs, method='Nelder-Mead')
    depth = res.x
    
    carLabel['x'] = depth*normCenter3D[0]
    carLabel['y'] = depth*normCenter3D[1] + carLabel['height']/2
    carLabel['z'] = depth

    return carLabel

def functionToMinimize(depth, calib3x4, carBox, carLabel, normCenter3D):
    carLabel['x'] = depth*normCenter3D[0]
    carLabel['y'] = depth*normCenter3D[1] + carLabel['height']/2
    carLabel['z'] = depth

    projected3DCarBox = getProjectedBox(calib3x4, carLabel)
    IoU2D = getIoU2D(carBox, projected3DCarBox)

    return 1-IoU2D

def getIoU2D(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    xminInter = max(xmin1, xmin2)
    yminInter = max(ymin1, ymin2)
    xmaxInter = min(xmax1, xmax2)
    ymaxInter = min(ymax1, ymax2)

    area1 = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)
    area2 = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)

    areaInter = max(0, xmaxInter - xminInter + 1) * max(0, ymaxInter - yminInter + 1)
    IoU2D = float(areaInter) / (area1 + area2 - areaInter)

    return IoU2D

def getProjectedBox(calib3x4, carLabel):
    corners2D = computeBox3D(carLabel, calib3x4)
    
    left = min(corners2D[0])
    top = min(corners2D[1])
    right = max(corners2D[0])
    bottom = max(corners2D[1])
    
    return [left, top, right, bottom]

''' =================================================================================== '''
''' ------------------------- Compute for every sampled image ------------------------- '''
''' =================================================================================== '''
with torch.no_grad():
    for sequenceIndex in random.sample(range(21), k=args.numberOfSamples):
        sequenceImagesDir = os.path.join(imageDir, '{:04d}'.format(sequenceIndex))
        imageList = sorted(glob.glob(os.path.join(sequenceImagesDir, '*.png')))

        for i, imagePath in enumerate(imageList):
            image = Image.open(imagePath)

            imageSize = np.array(image).shape[:-1]
            figSize = (args.imageScaleFactor*imageSize[1], args.imageScaleFactor*imageSize[0])
            fig, ax = plt.subplots(figsize=figSize)
            ax.axis('off')
            ax.imshow(image)

            calib3x4 = readCalibrationFile(calibDir, 2, sequenceIndex)
            invCalib3x3 = np.linalg.inv(calib3x4[:, :-1])

            imageTensor = detectionTransform(image.convert('RGB')).unsqueeze(0).to(device)
            detections = maskRCNN(imageTensor)

            boxes = detections[0]['boxes'].tolist()
            labels = detections[0]['labels'].tolist()
            scores = detections[0]['scores'].tolist()

            for detectionIndex in range(len(labels)):
                if scores[detectionIndex] < args.minimumScore:
                    break

                if labels[detectionIndex] == 3:
                    carBox = boxes[detectionIndex]
                    carImage = image.crop(carBox)
                    carImageTensor = angleTransform(carImage).unsqueeze(0).to(device)

                    with torch.no_grad():
                        localAngle = angleModel(carImageTensor).item()

                    rayAngle = getRayAngle(invCalib3x3, *carBox)
                    globalAngle = localAngle - rayAngle

                    carLabel = getDefaultLabel()
                    carLabel['type'] = 'Car'

                    carLabel['left'] = carBox[0]
                    carLabel['top'] = carBox[1]
                    carLabel['right'] = carBox[2]
                    carLabel['bottom'] = carBox[3]

                    carLabel['height'] = args.boxHeight
                    carLabel['length'] = args.boxLength
                    carLabel['width'] = args.boxWidth

                    carLabel['alpha'] = localAngle
                    carLabel['ry'] = globalAngle
                    
                    carLabel = boxFitOptimization(calib3x4, carBox, carLabel)
                    drawBox3D(ax, carLabel, calib3x4)
                    
            savePath = os.path.join(videosDir, '{:06d}.png'.format(i))
            fig.savefig(savePath, facecolor='k', transparent=True)
            plt.close()

        os.chdir(videosDir)
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'panic', '-framerate', '10', '-i', '%06d.png', '{:d}.mp4'.format(sequenceIndex)])
        [os.remove(image) for image in glob.glob('*.png')]
        
        
