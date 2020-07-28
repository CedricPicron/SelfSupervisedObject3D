import math
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import torch

from matplotlib.lines import Line2D
from PIL import Image
from tqdm import tqdm

''' ================================================================================ '''
''' ------------------------- Functions to assist in Kitti ------------------------- '''
''' ================================================================================ '''

def computeBox3D(objectLabel, P):
    h = objectLabel['height']
    l = objectLabel['length'] 
    w = objectLabel['width']

    xCorners = np.array([[l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]])
    yCorners = np.array([[0,0,0,0,-h,-h,-h,-h]])
    zCorners = np.array([[w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]])
    points3D = np.concatenate((xCorners, yCorners, zCorners), axis=0)

    ry = objectLabel['ry']
    R = np.array([[+math.cos(ry), 0, +math.sin(ry)], \
                  [      0      , 1,       0      ], \
                  [-math.sin(ry), 0, +math.cos(ry)]])

    corners3D = np.matmul(R, points3D)
    corners3D[0, :] = corners3D[0, :] + objectLabel['x']
    corners3D[1, :] = corners3D[1, :] + objectLabel['y']
    corners3D[2, :] = corners3D[2, :] + objectLabel['z']

    corners2D = projectPointsToImage(P, corners3D)

    return corners2D

def computeOrientation3D(objectLabel, P):
    ry = objectLabel['ry']
    R = np.array([[+math.cos(ry), 0, +math.sin(ry)], \
                  [      0      , 1,       0      ], \
                  [-math.sin(ry), 0, +math.cos(ry)]])

    xOrientation = np.array([[0.0, objectLabel['length']]])
    yOrientation = np.array([[0.0, 0.0]])
    zOrientation = np.array([[0.0, 0.0]])
    orientation3D = np.concatenate((xOrientation, yOrientation, zOrientation), axis=0)
    
    orientation3D = np.matmul(R, orientation3D)
    orientation3D[0, :] = orientation3D[0, :] + objectLabel['x']
    orientation3D[1, :] = orientation3D[1, :] + objectLabel['y']
    orientation3D[2, :] = orientation3D[2, :] + objectLabel['z']

    orientation2D = projectPointsToImage(P, orientation3D)

    return orientation2D

def displayFailureCases(criterion, experimentDir, gtLabelsDict, imageDir, outputLabelsDict, calibList=None, dim='2D', num=10):
    failureDir = '%s/%s' % (experimentDir, 'failure')
    subprocess.run(['mkdir', '-p', failureDir])

    errorList = []
    objectIds = []

    for counter, sequenceId in enumerate(gtLabelsDict.keys()):
        gtLabels = gtLabelsDict[key]
        outputLabels = outputLabelsDict[key]

        for detectionNumber in range(len(gtLabels)):
            gtLabel = gtLabels[detectionNumber]
            outputLabel = outputLabels[detectionNumber]

            error = criterion(gtLabel, outputLabel)
            errorList.append(error)

            objectId = [counter, detectionNumber, sequenceId]
            objectIds.append(objectId)

    errorList = np.array(errorList)
    sortedIndices = np.flip(np.argsort(errorList))
    maxIndices = sortedIndices[:num]

    for i in range(num):
        index = maxIndices[i]
        counter, detectionNumber, sequenceId = objectIds[index]

        imageName = '%s/%04d/%06d.png' % (imageDir, sequenceId, detectionNumber)
        image = Image.open(imageName)

        imageSize = np.array(image).shape[:-1]
        scaleFactor = 0.03

        figSize = (scaleFactor*imageSize[1], scaleFactor*imageSize[0])        
        fig, ax = plt.subplots(figsize=figSize)
        ax.imshow(image)

        gtLabel = gtLabelsDict[sequenceId][detectionNumber]
        outputLabel = outputLabelsDict[sequenceId][detectionNumber]

        gtString = labelToString(gtLabel, 'Groundtruth label')
        outputString = labelToString(outputLabel, 'Output label')

        filePath = '%s/%d.txt' % (failureDir, i)
        file = open(filePath, 'w')
        file.write(gtString)
        file.write(outputString)
        file.close()

        if dim == '2D':
            drawBox2D(ax, gtLabel, color='b')
            drawBox2D(ax, outputLabel, color='m')

        elif dim == '3D':
            P = calibList[counter]
            drawBox3D(ax, gtLabel, P, color='b')
            drawBox3D(ax, outputLabel, P, color='m')

        savePath = '%s/%d.png' % (failureDir, i)
        fig.savefig(savePath)
        plt.close()

def drawBox2D(ax, objectLabel, color=None):
    from matplotlib.patches import Rectangle

    left = objectLabel['left']
    top = objectLabel['top']
    width = objectLabel['right'] - left
    height = objectLabel['bottom'] - top

    if objectLabel['type'] != 'DontCare':
        color = getColor(color, objectLabel['occluded'])
        lineStyle = getLineStyle(objectLabel['truncated'])

        rectangle = Rectangle((left, top), width, height, ec=color, fill=False, ls=lineStyle, lw=3)
        ax.add_patch(rectangle)

        labelText = '%s\n%.1f°' % (objectLabel['type'], objectLabel['alpha']/math.pi * 180.0)
        xPosition = left + width/2
        yPosition = top - 15
        ax.text(xPosition, yPosition, labelText, backgroundcolor='k', color=color, ha='center', size=15, va='bottom', weight='bold')

    else:
        rectangle = Rectangle((left, top), width, height, ec='c', fill=False, ls='-', lw=2)
        ax.add_patch(rectangle)

def drawBox3D(ax, objectLabel, P, color=None):
    if objectLabel['type'] != 'DontCare':
        corners2D = computeBox3D(objectLabel, P)
        orientation2D = computeOrientation3D(objectLabel, P)

        frontIndices = np.array([[0, 1, 5, 4]])
        leftIndices = np.array([[1, 2, 6, 5]])
        backIndices = np.array([[2, 3, 7, 6]])
        rightIndices = np.array([[3, 0, 4, 7]])
        faceIndicesList = np.concatenate((frontIndices, leftIndices, backIndices, rightIndices), axis=0)

        color = getColor(color, objectLabel['occluded'])
        lineStyle = getLineStyle(objectLabel['truncated'])

        if corners2D is not None:
            for faceIndices in faceIndicesList:
                ax.add_line(Line2D(corners2D[0,faceIndices], corners2D[1,faceIndices], color=color, ls=lineStyle, lw=3))

        if orientation2D is not None:
            ax.add_line(Line2D(orientation2D[0,:], orientation2D[1,:], color='w', lw=4))
            ax.add_line(Line2D(orientation2D[0,:], orientation2D[1,:], color='k', lw=2))

def flipAlpha(alpha):
    return normalizeAngle(np.sign(alpha)*math.pi - alpha)

def generateLabelLine(objectLabel):
    keyList = ['frame', 'id', 'type', 'truncated', 'occluded', \
               'alpha', 'left', 'top', 'right', 'bottom', \
               'height', 'width', 'length', \
               'x', 'y', 'z', 'ry', 'moving', 'score']
    lineList = [''] * len(keyList)

    for i, key in enumerate(keyList):
        if key in objectLabel:
            if key in ['frame', 'id']:
                lineList[i] = '%d' % (objectLabel[key])
            elif key == 'type':
                lineList[i] = objectLabel[key]
            elif key in ['truncated', 'occluded', 'moving']:
                lineList[i] = '%d' % objectLabel[key]
            elif key in ['alpha', 'ry']:
                lineList[i] = '%f' % normalizeAngle(objectLabel[key])
            else:
                lineList[i] = '%f' % objectLabel[key]

        elif key in ['alpha', 'ry']:
            lineList[i] = '-10'
        elif key in ['x', 'y', 'z']:
            lineList[i] = '-1000'
        elif key in ['truncated', 'occluded', 'height', 'width', 'length']:
            lineList[i] = '-1'
        elif key in ['moving', 'score']:
            break
        
        else:
            raise Exception("The '%s' entry must be provided and was missing." % (key))
            
    line = ' '.join(lineList) + '\n'

    return line

def getColor(color, occlusionLevel):
    if color is not None:
        return color

    occlusionColors = ['g', 'y', 'r', 'w']
    occlusionColor = occlusionColors[occlusionLevel]

    return occlusionColor

def getDefaultLabel():
    objectLabel = {}

    objectLabel['frame'] = 0
    objectLabel['id'] = 0

    objectLabel['type'] = 'DontCare'
    objectLabel['truncated'] = 0.0
    objectLabel['occluded'] = 0
    objectLabel['alpha'] = 0.0

    objectLabel['left'] = 100.0
    objectLabel['top'] = 100.0
    objectLabel['right'] = 200.0
    objectLabel['bottom'] = 200.0
    
    objectLabel['height'] = 1.0
    objectLabel['width'] = 1.0
    objectLabel['length'] = 1.0

    objectLabel['x'] = 0.0
    objectLabel['y'] = 0.0
    objectLabel['z'] = 10.0

    objectLabel['ry'] = 0.0
    objectLabel['score'] = 0.0

    return objectLabel

def getLineStyle(truncationLevel):
    truncationLineStyles = ['-', '--']
    truncationIndex = int(truncationLevel>=0.1)
    lineStyle = truncationLineStyles[truncationIndex]

    return lineStyle

def getRayAngle(invCalibration3x3, left, top, right, bottom):
    xCenter = 0.5 * (left+right)
    yCenter = 0.5 * (top+bottom)
    center2D = np.array([xCenter, yCenter, 1])

    X, _, Z = invCalibration3x3.dot(center2D)
    rayAngle = math.acos( X/math.sqrt(X**2 + Z**2) ) - math.pi/2

    return rayAngle

def getRotationMatrix(objectLabel):
    ry = objectLabel['ry']
    R = np.array([[+math.cos(ry), 0, +math.sin(ry)], \
                  [      0      , 1,       0      ], \
                  [-math.sin(ry), 0, +math.cos(ry)]])

    return R

def getSplitAll():
    trainDict = {}
    valDict = {}

    trainDict['indices'] = [i for i in range(21)]
    valDict['indices'] = [i for i in range(21)]

    trainDict['readMode'] = 'Full'
    valDict['readMode'] = 'Full'

    return trainDict, valDict

def getSplitComplementary():
    trainDict = {}
    valDict = {}

    trainDict['indices'] = [i for i in range(21) if i not in [1, 2, 6, 20]]
    valDict['indices'] = [20]   

    trainDict['readMode'] = 'Full'
    valDict['readMode'] = 'Full'

    return trainDict, valDict

def getSplitHalves():
    trainDict = {}
    valDict = {}

    trainDict['indices'] = list(range(21))
    valDict['indices'] = list(range(21))

    trainDict['readMode'] = 'FirstHalf'
    valDict['readMode'] = 'SecondHalf'

    return trainDict, valDict

def getSplitVirtualKitti():
    trainDict = {}
    valDict = {}

    trainDict['indices'] = [1, 2, 6]
    valDict['indices'] = [20]

    trainDict['readMode'] = 'Full'
    valDict['readMode'] = 'Full'
    
    return trainDict, valDict

def labelToString(objectLabel, title):
    symbolLength = 38
    objectString = '=' * symbolLength + '\n'

    whiteSpace = ' ' * ((symbolLength-len(title))//2)
    objectString += whiteSpace + title + whiteSpace + '\n'
    objectString += '=' * symbolLength + '\n\n'

    objectString += 'Frame:        %d\n' % (objectLabel['frame'])
    objectString += 'Id:           %d\n\n' % (objectLabel['id'])

    objectString += 'Type:         %s\n' % (objectLabel['type'])
    objectString += 'Truncated:    %f\n' % (objectLabel['truncated'])
    objectString += 'Occluded:     %d\n\n' % (objectLabel['occluded'])

    objectString += 'Left edge:    %f\n' % (objectLabel['left'])
    objectString += 'Top edge:     %f\n' % (objectLabel['top'])
    objectString += 'Right edge:   %f\n' % (objectLabel['right'])
    objectString += 'Bottom edge:  %f\n\n' % (objectLabel['bottom'])

    objectString += 'X-coordinate: %f\n' % (objectLabel['x'])
    objectString += 'Y-coordinate: %f\n' % (objectLabel['y'])
    objectString += 'Z-coordinate: %f\n\n' % (objectLabel['z'])

    objectString += 'Height:       %f\n' % (objectLabel['height'])
    objectString += 'Length:       %f\n' % (objectLabel['length'])
    objectString += 'Width:        %f\n\n' % (objectLabel['width'])

    objectString += 'Local angle:  %.2f\n' % (normalizeAngle(objectLabel['alpha'])/math.pi * 180.0)
    objectString += 'Yaw angle:    %.2f\n\n' % (normalizeAngle(objectLabel['ry'])/math.pi * 180.0)

    if 'score' in objectLabel:
        objectString += 'Score:        %f\n\n' % (objectLabel['score'])

    return objectString

def normalizeAngle(angle, period=2*math.pi):
    angle = angle % period

    if angle <= -period/2:
        angle += period
    elif angle > period/2:
        angle -= period

    return angle

def normalizeAngleArray(array, period=2*math.pi):
    array = np.mod(array, period)
    
    array = np.where(array <= -period/2, array+period, array)
    array = np.where(array > period/2, array-period, array)

    return array

def normalizeAngleTensor(tensor, period=2*math.pi):
    tensor = torch.fmod(tensor, period)
    
    tensor = torch.where(tensor <= -period/2, tensor+period, tensor)
    tensor = torch.where(tensor > period/2, tensor-period, tensor)

    return tensor

def projectPointsToImage(P, points3D):
    points3D = np.concatenate((points3D, np.ones((1, points3D.shape[1]))), axis=0)

    points2D = np.matmul(P, points3D)
    points2D = points2D[:-1]/points2D[2]

    return points2D

def projectPointToImage(P, point3D):
    point3D = np.append(point3D, 1)

    point2D = P.dot(point3D)
    point2D = point2D[:-1]/point2D[2]

    return point2D

def readCalibrationFile(calibrationDir, cameraNumber, sequenceId):
    fileName = '%s/%04d.txt' % (calibrationDir, sequenceId)
    file = open(fileName, 'r')

    lines = file.read().splitlines()
    line = lines[cameraNumber].split(' ')[1:]

    flatP = np.array([float(i) for i in line if i])
    P = np.reshape(flatP, (3, 4))

    file.close()
    return P

def readCalibrationFiles(calibrationDir, cameraNumber, sequenceIds):
    calibrationDict = {}

    for sequenceId in sequenceIds:
        calibration = readCalibrationFile(calibrationDir, cameraNumber, sequenceId)
        calibrationDict[sequenceId] = calibration

    return calibrationDict

def readLabelFile(labelDir, sequenceId, readMode='Full'):
    fileName = '%s/%04d.txt' % (labelDir, sequenceId)
    objectLabels = []
    
    if os.stat(fileName).st_size:
        with open(fileName, 'r') as file:
            if readMode in ['FirstHalf', 'Full', 'SecondHalf']:
                lines = file.read().splitlines()
                numberOfFrames = int(lines[-1].split(' ')[0]) + 1
            
            for line in lines:
                objectLabel = readLabelLine(line)

                condition1 = readMode == 'FirstHalf' and objectLabel['frame'] >= numberOfFrames//2
                condition2 = readMode == 'SecondHalf' and objectLabel['frame'] < numberOfFrames//2
                    
                if not condition1 and not condition2:
                    objectLabels.append(objectLabel)

    return objectLabels

def readLabelFiles(labelDir, sequenceIds, readMode='Full'):
    sequenceLabelsDict = {}

    for sequenceId in sequenceIds:
        objectLabels = readLabelFile(labelDir, sequenceId, readMode=readMode)
        sequenceLabelsDict[sequenceId] = objectLabels

    return sequenceLabelsDict

def readLabelLine(line):
    line = line.split(' ')
    objectLabel = {}

    objectLabel['frame'] = int(line[0])
    objectLabel['id'] = int(line[1])
    objectLabel['type'] = line[2]
    objectLabel['truncated'] = int(line[3])
    objectLabel['occluded'] = int(line[4])
    objectLabel['alpha'] = float(line[5])
    objectLabel['left'] = float(line[6])
    objectLabel['top'] = float(line[7])
    objectLabel['right'] = float(line[8])
    objectLabel['bottom'] = float(line[9])
    objectLabel['height'] = float(line[10])
    objectLabel['width'] = float(line[11])
    objectLabel['length'] = float(line[12])
    objectLabel['x'] = float(line[13])
    objectLabel['y'] = float(line[14])
    objectLabel['z'] = float(line[15])
    objectLabel['ry'] = float(line[16])

    try:
        objectLabel['moving'] = bool(int(line[17]))
        objectLabel['score'] = float(line[18])

    except:
        pass

    return objectLabel

def runOfflineEvaluation(labelDir, sequenceLabelsDict, writeDir):
    dataDir = '%s/%s' % (writeDir, 'data')
    subprocess.run(['mkdir', '-p', dataDir])
    writeLabelFiles(sequenceLabelsDict, dataDir)

    command = 'python evaluate_tracking_offline.py %s %s' % (labelDir, writeDir)
    cwd = '/esat/ruchba/cpicron/Datasets/Kitti/Tracking/devkit_tracking/devkit/python'
    subprocess.run(command, shell=True, cwd=cwd)

def writeLabelFile(objectLabels, sequenceId, writeDir):
    fileName = '%s/%04d.txt' % (writeDir, sequenceId)
    file = open(fileName, 'w')

    for objectLabel in objectLabels:
        line = generateLabelLine(objectLabel)
        file.write(line)
    
    file.close()

def writeLabelFiles(sequenceLabelsDict, writeDir, zipName=''):
    subprocess.run('rm -rf *.txt', shell=True, cwd=writeDir)

    for sequenceId, objectLabels in sequenceLabelsDict.items():
        writeLabelFile(objectLabels, sequenceId, writeDir)

    if zipName:
        subprocess.run('zip -mq %s *.txt' % (zipName), shell=True, cwd=writeDir)

''' ============================================================= '''
''' ------------------------- Test code ------------------------- '''
''' ============================================================= '''

def main():

    ''' =============================================================== '''
    ''' ------------------------- Test splits ------------------------- '''
    ''' =============================================================== '''

    trainDictComp, valDictComp = getSplitComplementary()
    trainDictHalf, valDictHalf = getSplitHalves()
    trainDictVirt, valDictVirt = getSplitVirtualKitti()

    print('Number of train and val sequences: %d | %d' % (len(trainDictComp['indices']), len(valDictComp['indices'])))
    print('Number of train and val sequences: %d | %d' % (len(trainDictHalf['indices']), len(valDictHalf['indices'])))
    print('Number of train and val sequences: %d | %d' % (len(trainDictVirt['indices']), len(valDictVirt['indices'])))

    ''' =================================================================== '''
    ''' ------------------------- Test read/write ------------------------- '''
    ''' =================================================================== '''

    labelDir = '/esat/ruchba/cpicron/Datasets/Kitti/Tracking/training/label_02'
    readMode = trainDictHalf['readMode']
    sequenceIds = trainDictHalf['indices']
    writeDir = '/esat/ruchba/cpicron/Datasets/Kitti/Tracking/utils/Testing'
    zipName = 'copy.zip'

    sequenceLabelsDict = readLabelFiles(labelDir, sequenceIds, readMode=readMode)

    for objectLabels in sequenceLabelsDict.values():
        for objectLabel in objectLabels:
            objectLabel['score'] = 0.5

    writeDirWithZip = '%s/%s' % (writeDir, 'WriteLabelFiles/WithZip')    
    writeDirWithoutZip = '%s/%s' % (writeDir, 'WriteLabelFiles/WithoutZip/data/')    

    writeLabelFiles(sequenceLabelsDict, writeDirWithoutZip)
    writeLabelFiles(sequenceLabelsDict, writeDirWithZip, zipName)

    ''' ====================================================================== '''
    ''' ------------------------- Test visualization ------------------------- '''
    ''' ====================================================================== '''

    calibDir = '/esat/ruchba/cpicron/Datasets/Kitti/Tracking/training/calib'
    cameraNumber = 2

    calibrationDict = readCalibrationFiles(calibDir, cameraNumber, sequenceIds)
    imageDir = '/esat/ruchba/cpicron/Datasets/Kitti/Tracking/training/image_02'

    sequenceId = 0
    objectLabels = sequenceLabelsDict[sequenceId]
    P = calibrationDict[sequenceId]

    for frameId in tqdm(range(20)):
        imageName = '%s/%04d/%06d.png' % (imageDir, sequenceId, frameId)
        image = Image.open(imageName)

        imageSize = np.array(image).shape[:-1]
        scaleFactor = 0.03

        figSize = (scaleFactor*imageSize[1], scaleFactor*imageSize[0])        
        fig1, ax1 = plt.subplots(figsize=figSize)
        fig2, ax2 = plt.subplots(figsize=figSize)

        ax1.imshow(image)
        ax2.imshow(image)

        for objectLabel in objectLabels:
            if objectLabel['frame'] == frameId:
                drawBox2D(ax1, objectLabel)
                drawBox3D(ax2, objectLabel, P)

        savePath2D = '%s/Visualization/Draw2D/%d.png' % (writeDir, frameId)
        savePath3D = '%s/Visualization/Draw3D/%d.png' % (writeDir, frameId)

        fig1.savefig(savePath2D)
        fig2.savefig(savePath3D)
        plt.close('all')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass



