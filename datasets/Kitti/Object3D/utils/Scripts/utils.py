import math
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess

from matplotlib.lines import Line2D
from PIL import Image

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

def displayFailureCases(criterion, experimentDir, gtLabelsDict, imageDir, outputLabelsDict, calibDict=None, dim='2D', num=10):
    failureDir = '%s/%s' % (experimentDir, 'failure')
    os.makedirs(failureDir, exist_ok=True)

    errorList = []
    objectIds = []

    for imageId in gtLabelsDict.keys():
        gtLabels = gtLabelsDict[imageId]
        outputLabels = outputLabelsDict[imageId]

        for objectNumber in range(len(gtLabels)):
            gtLabel = gtLabels[objectNumber]
            outputLabel = outputLabels[objectNumber]

            error = criterion(gtLabel, outputLabel)
            errorList.append(error)

            objectId = [imageId, objectNumber]
            objectIds.append(objectId)

    errorList = np.array(errorList)
    sortedIndices = np.flip(np.argsort(errorList))
    maxIndices = sortedIndices[:num]

    for i in range(num):
        index = maxIndices[i]
        imageId, objectNumber = objectIds[index]

        imageName = '%s/%06d.png' % (imageDir, imageId)
        image = Image.open(imageName)

        imageSize = np.array(image).shape[:-1]
        scaleFactor = 0.03

        figSize = (scaleFactor*imageSize[1], scaleFactor*imageSize[0])        
        fig, ax = plt.subplots(figsize=figSize)
        ax.imshow(image)

        gtLabel = gtLabelsDict[imageId][objectNumber]
        outputLabel = outputLabelsDict[imageId][objectNumber]

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
            P = calibDict[imageId]
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

def generateLabelLine(objectLabel):
    keyList = ['type', 'truncated', 'occluded', 'alpha', \
               'left', 'top', 'right', 'bottom', \
               'height', 'width', 'length', \
               'x', 'y', 'z', 'ry', 'score']
    lineList = [''] * len(keyList)

    for i, key in enumerate(keyList):
        if key in objectLabel:
            if key == 'type':
                lineList[i] = objectLabel[key]
            elif key == 'occluded':
                lineList[i] = '%.d' % objectLabel[key]
            elif key in ['alpha', 'ry']:
                lineList[i] = '%.2f' % normalizeAngle(objectLabel[key])
            else:
                lineList[i] = '%.2f' % objectLabel[key]

        elif key in ['alpha', 'ry']:
            lineList[i] = '-10'
        elif key in ['x', 'y', 'z']:
            lineList[i] = '-1000'
        elif key in ['truncated', 'occluded', 'height', 'width', 'length']:
            lineList[i] = '-1'
        
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

def getIndicesSplit1():
    ''' Files from https://github.com/garrickbrazil/M3D-RPN/tree/master/data/kitti_split1 (03/20/2020) '''

    trainFileName = '/esat/ruchba/cpicron/Datasets/Kitti/Object3D/devkit_object/splits/split1/train.txt'
    valFileName = '/esat/ruchba/cpicron/Datasets/Kitti/Object3D/devkit_object/splits/split1/val.txt'

    trainFile = open(trainFileName, 'r')
    valFile = open(valFileName, 'r')

    trainLines = trainFile.read().splitlines()
    valLines = valFile.read().splitlines()

    train1 = [int(s) for s in trainLines]
    val1 = [int(s) for s in valLines]

    return train1, val1

def getIndicesSplit2():
    ''' From from https://github.com/garrickbrazil/M3D-RPN/tree/master/data/kitti_split2 (03/20/2020) '''

    import scipy.io
    fileName = '/esat/ruchba/cpicron/Datasets/Kitti/Object3D/devkit_object/splits/split2/kitti_ids_new.mat'
    mat = scipy.io.loadmat(fileName)

    train2 = mat['ids_train'][0]
    val2 = mat['ids_val'][0]

    return train2, val2

def getIndicesTrainVal():
    return list(range(7481))

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

def labelToString(objectLabel, title):
    symbolLength = 38
    objectString = '=' * symbolLength + '\n'

    whiteSpace = ' ' * ((symbolLength-len(title))//2)
    objectString += whiteSpace + title + whiteSpace + '\n'
    objectString += '=' * symbolLength + '\n\n'

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

def readCalibrationFile(calibrationDir, cameraNumber, imageId):
    fileName = '%s/%06d.txt' % (calibrationDir, imageId)
    file = open(fileName, 'r')

    lines = file.read().splitlines()
    line = lines[cameraNumber].split(' ')[1:]

    flatP = np.array([float(i) for i in line])
    P = np.reshape(flatP, (3, 4))

    file.close()
    return P

def readCalibrationFiles(calibrationDir, cameraNumber, imageIds):
    calibrationDict = {}

    for imageId in imageIds:
        calibration = readCalibrationFile(calibrationDir, cameraNumber, imageId)
        calibrationDict[imageId] = calibration

    return calibrationDict

def readLabelFile(imageId, labelDir):
    fileName = '%s/%06d.txt' % (labelDir, imageId)
    file = open(fileName, 'r')

    lines = file.read().splitlines()
    objectLabels = []

    for line in lines:
        objectLabel = readLabelLine(line)
        objectLabels.append(objectLabel)

    file.close()
    return objectLabels

def readLabelFiles(imageIds, labelDir):
    imageLabelsDict = {}

    for imageId in imageIds:
        objectLabels = readLabelFile(imageId, labelDir)
        imageLabelsDict[imageId] = objectLabels

    return imageLabelsDict

def readLabelLine(line):
    line = line.split(' ')
    objectLabel = {}

    objectLabel['type'] = line[0]
    objectLabel['truncated'] = float(line[1])
    objectLabel['occluded'] = int(line[2])
    objectLabel['alpha'] = float(line[3])
    objectLabel['left'] = float(line[4])
    objectLabel['top'] = float(line[5])
    objectLabel['right'] = float(line[6])
    objectLabel['bottom'] = float(line[7])
    objectLabel['height'] = float(line[8])
    objectLabel['width'] = float(line[9])
    objectLabel['length'] = float(line[10])
    objectLabel['x'] = float(line[11])
    objectLabel['y'] = float(line[12])
    objectLabel['z'] = float(line[13])
    objectLabel['ry'] = float(line[14])

    return objectLabel

def runOfflineEvaluation(labelDir, imageLabelsDict, writeDir):
    dataDir = '%s/%s' % (writeDir, 'data')
    os.makedirs(dataDir, exist_ok=True)
    writeLabelFiles(imageLabelsDict, dataDir)

    command = './evaluate_object_offline %s %s' % (labelDir, writeDir)
    cwd = '/esat/ruchba/cpicron/Datasets/Kitti/Object3D/devkit_object/cpp'
    subprocess.run(command, shell=True, cwd=cwd)

def writeLabelFile(imageId, objectLabels, writeDir):
    fileName = '%s/%06d.txt' % (writeDir, imageId)
    file = open(fileName, 'w')

    for objectLabel in objectLabels:
        line = generateLabelLine(objectLabel)
        file.write(line)
    
    file.close()

def writeLabelFiles(imageLabelsDict, writeDir, zipName=''):
    subprocess.run('rm -rf *.txt', shell=True, cwd=writeDir)

    for imageId, objectLabels in imageLabelsDict.items():
        writeLabelFile(imageId, objectLabels, writeDir)

    if zipName:
        subprocess.run('zip -mq %s *.txt' % (zipName), shell=True, cwd=writeDir)

''' ============================================================= '''
''' ------------------------- Test code ------------------------- '''
''' ============================================================= '''

def main():

    ''' =============================================================== '''
    ''' ------------------------- Test splits ------------------------- '''
    ''' =============================================================== '''

    trainval = getIndicesTrainVal()
    train1, val1 = getIndicesSplit1()
    train2, val2 = getIndicesSplit2()

    print('\nTrainVal length: %d' % len(trainval))
    print('Train1 length: %d' % len(train1))
    print('Val1 length: %d' % len(val1))
    print('Train2 length: %d' % len(train2))
    print('Val2 length: %d\n' % len(val2))

    ''' =================================================================== '''
    ''' ------------------------- Test read/write ------------------------- '''
    ''' =================================================================== '''

    imageIds = trainval
    labelDir = '/esat/ruchba/cpicron/Datasets/Kitti/Object3D/training/label_2'
    writeDir = '/esat/ruchba/cpicron/Datasets/Kitti/Object3D/utils/Testing'
    zipName = 'copy.zip'

    imageLabelsDict = readLabelFiles(imageIds, labelDir)

    for key in imageLabelsDict:
        objectLabels = imageLabelsDict[key]

        for objectLabel in objectLabels:
            objectLabel['score'] = 0.5

    writeDirWithZip = '%s/%s' % (writeDir, 'WriteLabelFiles/WithZip')    
    writeDirWithoutZip = '%s/%s' % (writeDir, 'WriteLabelFiles/WithoutZip/data/')    

    writeLabelFiles(imageLabelsDict, writeDirWithoutZip)
    writeLabelFiles(imageLabelsDict, writeDirWithZip, zipName)

    ''' ====================================================================== '''
    ''' ------------------------- Test visualization ------------------------- '''
    ''' ====================================================================== '''

    calibDir = '/esat/ruchba/cpicron/Datasets/Kitti/Object3D/training/calib'
    cameraNumber = 2
    calibrationDict = readCalibrationFiles(calibDir, cameraNumber, imageIds)
    imageDir = '/esat/ruchba/cpicron/Datasets/Kitti/Object3D/training/image_2'

    for imageIndex in range(20):
        objectLabels = imageLabelsDict[imageIndex]
        P = calibrationDict[imageIndex]

        imageName = '%s/%06d.png' % (imageDir, imageIndex)
        image = Image.open(imageName)

        imageSize = np.array(image).shape[:-1]
        scaleFactor = 0.03

        figSize = (scaleFactor*imageSize[1], scaleFactor*imageSize[0])        
        fig1, ax1 = plt.subplots(figsize=figSize)
        fig2, ax2 = plt.subplots(figsize=figSize)

        ax1.imshow(image)
        ax2.imshow(image)

        for objectLabel in objectLabels:
            drawBox2D(ax1, objectLabel)
            drawBox3D(ax2, objectLabel, P)

        savePath2D = '%s/Visualization/Draw2D/%d.png' % (writeDir, imageIndex)
        savePath3D = '%s/Visualization/Draw3D/%d.png' % (writeDir, imageIndex)

        fig1.savefig(savePath2D)
        fig2.savefig(savePath3D)
        plt.close('all')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass



