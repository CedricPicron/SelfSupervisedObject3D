import sys
sys.path.insert(1, '../../../../datasets/Kitti/Object3D/utils/Scripts')

from utils import computeBox3D
from utils import getDefaultLabel
from utils import getIndicesTrainVal
from utils import getRayAngle
from utils import normalizeAngle
from utils import readCalibrationFile
from utils import readLabelFile
from utils import runOfflineEvaluation

import argparse
import math
import numpy as np
import os
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
    default='../../../../datasets/Kitti/Object3D/training',
    help='Path to dataset location.')
argparser.add_argument(
    '--epsilon',
    metavar='THRESHOLD',
    type=float,
    default=0.5,
    help='Minimum overlap (2D IoU) of estimated box with gt. box required for postive assignment.')
argparser.add_argument(
    '--experimentNumber',
    metavar='N',
    type=int,
    default=1,
    help='Number corresponding to experiment.')
argparser.add_argument(
    '--experimentRoot',
    metavar='PATH',
    type=str,
    default='..',
    help='Path to experiment base directory.')
args = argparser.parse_args()

''' ================================================================== '''
''' ------------------------- Initialization ------------------------- '''
''' ================================================================== '''

calibDir = os.path.join(args.dataRoot, 'calib')
imageDir = os.path.join(args.dataRoot, 'image_2')

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

''' =========================================================================== '''
''' ------------------------- Compute for every image ------------------------- '''
''' =========================================================================== '''

imageIndices = getIndicesTrainVal()
imageLabels = {imageIndex: [] for imageIndex in imageIndices}

for imageIndex in imageIndices:
    calib3x4 = readCalibrationFile(calibDir, 2, imageIndex)
    invCalib3x3 = np.linalg.inv(calib3x4[:, :-1])

    imagePath = os.path.join(imageDir, '{:06d}.png'.format(imageIndex))
    image = Image.open(imagePath).convert('RGB')
    imageTensor = detectionTransform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        detections = maskRCNN(imageTensor)

    boxes = detections[0]['boxes'].tolist()
    labels = detections[0]['labels'].tolist()
    scores = detections[0]['scores'].tolist()

    for detectionIndex in range(len(labels)):
        if labels[detectionIndex] == 3:
            carBox = boxes[detectionIndex]
            carScore = scores[detectionIndex]

            carImage = image.crop(carBox)
            carImageTensor = angleTransform(carImage).unsqueeze(0).to(device)

            with torch.no_grad():
                localAngle = angleModel(carImageTensor).item()

            rayAngle = getRayAngle(invCalib3x3, *carBox)
            globalAngle = localAngle - rayAngle

            carLabel = getDefaultLabel()
            carLabel['type'] = 'Car'
            carLabel['score'] = carScore

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
            imageLabels[imageIndex].append(carLabel)

''' ========================================================================== '''
''' ------------------------- Run offline evaluation ------------------------- '''
''' ========================================================================== '''

labelDir = os.path.join(args.dataRoot, 'label_2')
resultDir = os.path.join(args.experimentRoot, 'Results/Optimization3D', str(args.experimentNumber))
runOfflineEvaluation(labelDir, imageLabels, resultDir)

errorKeys = ['height', 'length', 'width', 'x', 'y', 'z', 'ry']
errorDict = {errorKey: [] for errorKey in errorKeys}

for imageIndex, carLabels in imageLabels.items():
    gtLabels = readLabelFile(imageIndex, labelDir)
    gtLabels = [gtLabel for gtLabel in gtLabels if gtLabel['type'] == 'Car']
    gtAvailable = [True for _ in range(len(gtLabels))]

    for carLabel in carLabels:
        carBox2D = (carLabel['left'], carLabel['top'], carLabel['right'], carLabel['bottom'])
        bestOverlap = args.epsilon

        for gtIndex, gtLabel in enumerate(gtLabels):
            if gtAvailable[gtIndex]:
                gtBox2D = (gtLabel['left'], gtLabel['top'], gtLabel['right'], gtLabel['bottom'])
                overlap = getIoU2D(carBox2D, gtBox2D)

                if overlap > bestOverlap:
                    bestIndex = gtIndex
                    bestOverlap = overlap

        if bestOverlap > args.epsilon:
            gtAvailable[bestIndex] = False
            gtLabel = gtLabels[bestIndex]
            
            errorDict['height'].append(abs(carLabel['height']-gtLabel['height']))
            errorDict['length'].append(abs(carLabel['length']-gtLabel['length']))
            errorDict['width'].append(abs(carLabel['width']-gtLabel['width']))
            errorDict['x'].append(abs(carLabel['x']-gtLabel['x']))
            errorDict['y'].append(abs(carLabel['y']-gtLabel['y']))
            errorDict['z'].append(abs(carLabel['z']-gtLabel['z']))

            globalAngleError = abs(normalizeAngle(carLabel['ry']-gtLabel['ry']))
            errorDict['ry'].append(globalAngleError * 180.0/math.pi)

medianErrors = [np.median(errorDict[errorKey]) for errorKey in errorKeys]
meanErrors = [np.mean(errorDict[errorKey]) for errorKey in errorKeys]
outputFileName = os.path.join(resultDir, 'out.txt')

with open(outputFileName, 'w') as outputFile:
    for i, errorKey in enumerate(errorKeys):
        outputFile.write('%s error: %.2f | %.2f\n' % (errorKey.capitalize(), medianErrors[i], meanErrors[i]))

