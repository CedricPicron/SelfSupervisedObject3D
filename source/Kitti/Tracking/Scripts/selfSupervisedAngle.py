import sys
sys.path.insert(1, '../../../../datasets/Kitti/Tracking/utils/Scripts')

from utils import getSplitAll
from utils import getSplitComplementary
from utils import getSplitVirtualKitti
from utils import readLabelFiles

import argparse
import math
import numpy as np
import os
import random
import time
import torch

from PIL import Image
from PIL.ImageOps import mirror

from torch import nn
from torch.nn.functional import normalize
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from torchvision import transforms
from torchvision.models import resnext50_32x4d

''' ======================================================================== '''
''' ------------------------- Command-line options ------------------------- '''
''' ======================================================================== '''

argparser = argparse.ArgumentParser(description=__doc__)
argparser.add_argument(
    '--batchSize',
    metavar='BATCH',
    type=int,
    default=32,
    help='Mini-batch size.')
argparser.add_argument(
    '--cameraNumber',
    metavar='N',
    type=int,
    default=2,
    help='Number corresponding to Kitti camera.')
argparser.add_argument(
    '--dataRoot',
    metavar='PATH',
    type=str,
    default='../../../../datasets/Kitti/Tracking',
    help='Path to dataset location.')
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
argparser.add_argument(
    '--huberQuadSize',
    metavar='SIZE',
    type=float,
    default=20.0,
    help='Size of quadratic region in Huber loss.')
argparser.add_argument(
    '--labelNumber',
    metavar='NUMBER',
    type=int,
    choices=[2, 3, 7, 8],
    default=7,
    help='2: Gt. detection and angle differences. All cars.\n\
          3: Custom detection and gt. angle differences. All cars.\n\
          7: Custom detection and angle differences. All cars.\n\
          8: Gt. detection and custom angle differences. All cars.')
argparser.add_argument(
    '--learningRate',
    metavar='LR',
    type=float,
    default=2e-5,
    help='Initial learning rate.')
argparser.add_argument(
    '--loadModelNumber',
    metavar='N',
    type=int,
    default=0,
    help='Model number of model to load (relative load).')
argparser.add_argument(
    '--loadModelPath',
    metavar='PATH',
    type=str,
    default='',
    help='Path of model to load (absolute load).')
argparser.add_argument(
    '--loadModelRendering',
    metavar='MODE',
    type=str,
    choices=['', 'all', 'clone', 'fog', 'morning', 'overcast', 'rain', 'sunset'],
    default='all',
    help='If non-empty, load Virtual Kitti model trained on specified render mode.')
argparser.add_argument(
    '--minBbSize',
    metavar='SIZE',
    type=float,
    default=0.0,
    help='Minimum bounding box size used during training.')
argparser.add_argument(
   '--multiStepLrMilestones',
    metavar='MILESTONES',
    nargs='+',
    type=int,
    default=[20],
    help='Epochs at which learning rate is changed according to scheduler.')
argparser.add_argument(
    '--optimizer',
    metavar='OPTIM',
    type=str,
    choices=['SGD'],
    default='SGD',
    help='Optimizer used during training.')
argparser.add_argument(
    '--optimizerMomentum',
    metavar='MOMENTUM',
    type=float,
    default=0.9,
    help='Momentum factor used by optimizer.')
argparser.add_argument(
    '--pruneThreshold',
    metavar='PRUNE',
    type=float,
    default=1.0,
    help='Threshold determining which entries are pruned from sequence.')
argparser.add_argument(
    '--removeThreshold',
    metavar='REMOVE',
    type=float,
    default=1.0,
    help='Threshold determining which entries should be removed.')
argparser.add_argument(
    '--representation',
    metavar='REPR',
    type=str,
    choices=['Single', 'Double'],
    default='Single',
    help='Type of angle representation.')
argparser.add_argument(
    '--resume',
    action='store_true',
    help='Resume training from checkpoint.')
argparser.add_argument(
    '--scheduler',
    metavar='SCHED',
    type=str,
    choices=['MultiStepLR'],
    default='MultiStepLR',
    help='Scheduler type.')
argparser.add_argument(
    '--split',
    metavar='SPLIT',
    type=str,
    choices=['All', 'Complementary', 'Halves', 'VirtualKitti'],
    default='All',
    help='Train-validation split.')
argparser.add_argument(
    '--splitPercentage',
    metavar='PERCENT',
    type=float,
    default=0.8,
    help='Percentage of sequence belonging to training (All-split case only).')
argparser.add_argument(
    '--stepLrGamma',
    metavar='GAMMA',
    type=float,
    default=0.1,
    help='Step scheduler decay rate.')
argparser.add_argument(
    '--trainingCycles',
    metavar='CYCLES',
    type=int,
    default=1,
    help='Total number of training cycles.')
argparser.add_argument(
    '--trainingEpochs',
    metavar='EPOCHS',
    type=int,
    default=30,
    help='Total number of training epochs.')
argparser.add_argument(
    '--virtualModelsRoot',
    metavar='PATH',
    type=str,
    default='../../../VirtualKitti/Models',
    help='Path used to retrieve models trained on Virtual Kitti.')
argparser.add_argument(
    '--weightDecay',
    metavar='WEIGHT',
    type=float,
    default=1e-4,
    help='L2 weight decay factor.')
argparser.add_argument(
    '--workers',
    metavar='N',
    type=int,
    default=8,
    help='Number of workers.')

''' ================================================================================ '''
''' ------------------------- Collection of custom classes ------------------------- '''
''' ================================================================================ '''

class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, x):
        return nn.functional.normalize(x)

class SinCosToAngle(nn.Module):
    def __init__(self):
        super(SinCosToAngle, self).__init__()

    def forward(self, x):
        return torch.atan2(x[:,0], x[:,1])

class TrainDataset(Dataset):
    def __init__(self, angleModel, device, dictionary, percentage, root, transform):
        super(TrainDataset, self).__init__()
        
        self.root = root
        self.transform = transform

        printing = args.labelNumber in [2, 3] and args.trainingCycles == 1
        self.getDetections(dictionary, percentage)
        self.getSelfSupervisedTargets(angleModel, device, printing=printing)
        
    def getDetections(self, dictionary, percentage):
        self.boundingBoxes = []   
        self.mainIndicesDict = {}
        self.offsetTargets = []    
        self.toFrameIndexList = []
        self.toSequenceIndexList = []

        labelDir = '%s/label_0%d' % (self.root, args.labelNumber)
        sequenceIds = dictionary['indices']
        readMode = dictionary['readMode']
        sequenceLabelsDict = readLabelFiles(labelDir, sequenceIds, readMode=readMode)

        for sequenceIndex, objectLabels in sequenceLabelsDict.items():
            self.mainIndicesDict[sequenceIndex] = {}
            endIndex = int(len(objectLabels)*percentage)

            for labelNumber, objectLabel in enumerate(objectLabels[:endIndex]):
                if objectLabel['type'] == 'Car':
                    left = objectLabel['left']
                    top = objectLabel['top']
                    right = objectLabel['right']
                    bottom = objectLabel['bottom']

                    if (right-left)*(bottom-top) >= args.minBbSize:
                        self.boundingBoxes.append((left, top, right, bottom))
                        mainIndex = len(self.toFrameIndexList)                     
                        objectId = objectLabel['id']

                        if objectId in self.mainIndicesDict[sequenceIndex].keys():
                            self.mainIndicesDict[sequenceIndex][objectId].append(mainIndex)
                        else:
                            self.mainIndicesDict[sequenceIndex][objectId] = [mainIndex]

                        self.offsetTargets.append(objectLabel['alpha'])
                        self.toFrameIndexList.append(objectLabel['frame'])
                        self.toSequenceIndexList.append(sequenceIndex)

    def getOutputAngle(self, angleModel, device, mainIndex):
        frameIndex = self.toFrameIndexList[mainIndex]
        sequenceIndex = self.toSequenceIndexList[mainIndex]

        imagePath = '%s/training/image_0%d/%04d/%06d.png' % (args.dataRoot, args.cameraNumber, sequenceIndex, frameIndex)
        fullImage = Image.open(imagePath).convert('RGB')

        boundingBox = self.boundingBoxes[mainIndex]
        bbImage = fullImage.crop(boundingBox)
        
        bbImage = self.transform(bbImage)
        bbImage = bbImage.unsqueeze(0)
        bbImage = bbImage.to(device)

        with torch.set_grad_enabled(False):
            outputAngle = angleModel(bbImage)
            outputAngle = outputAngle.item()

        return outputAngle

    def getSelfSupervisedTargets(self, angleModel, device, printing=False):
        angleModel.eval()

        self.offsetTargets = np.array(self.offsetTargets)
        self.selfSupervisedTargets = []
        self.toMainIndexList = []

        for sequenceIndex in self.mainIndicesDict.keys():
            for objectId in self.mainIndicesDict[sequenceIndex].keys():
                objectMainIndices = self.mainIndicesDict[sequenceIndex][objectId]
                outputAngles = [0.0 for i in range(len(objectMainIndices))]

                for i, mainIndex in enumerate(objectMainIndices):
                    outputAngles[i] = self.getOutputAngle(angleModel, device, mainIndex)
    
                targetAngles = self.offsetTargets[objectMainIndices]
                objectTargets = self.getTargets(outputAngles, targetAngles, printing=printing)

                if len(objectTargets) > 0:
                    self.selfSupervisedTargets.extend(objectTargets)
                    self.toMainIndexList.extend(objectMainIndices)

    def getTargets(self, outputAngles, targetAngles, printing=False):
        N = len(outputAngles)

        mask = [True for i in range(N)]
        pruningFinished = False
        rankList = [0 for i in range(N)]

        for rankIndex in range(N-1, 1, -1):
            inconsistencyList = [0.0 for i in range(N)]    

            for index1 in range(N):
                if not mask[index1]:
                    continue

                outputAngle1 = outputAngles[index1]
                targetAngle1 = targetAngles[index1]

                for index2 in range(N):
                    if index1 == index2 or not mask[index2]:
                        continue

                    outputAngle2 = outputAngles[index2]
                    targetAngle2 = targetAngles[index2]

                    outputDifference = outputAngle2 - outputAngle1
                    targetDifference = targetAngle2 - targetAngle1

                    inconsistencyList[index1] += abs(outputDifference - targetDifference)

            for i, inconsistency in enumerate(inconsistencyList):
                if i == 0:
                    maxInconsistency = inconsistency
                    minInconsistency = inconsistency
                    maxIndex = i

                elif i > 0 and inconsistency > 0.0:
                    if inconsistency > maxInconsistency:
                        maxInconsistency = inconsistency
                        maxIndex = i

                    elif inconsistency < minInconsistency:
                        minInconsistency = inconsistency

                    if minInconsistency == 0.0:
                        minInconsistency = inconsistency

            if maxInconsistency/minInconsistency <= args.pruneThreshold and not pruningFinished:
                pruningFinished = True
                thresholdIndex = rankIndex

            mask[maxIndex] = False
            rankList[rankIndex] = maxIndex

        counter = 0
        for index in range(N):
            if mask[index]:
                rankList[counter] = index
                counter += 1

        if not pruningFinished:
            thresholdIndex = 1

        if printing:
            print('Output angles (top 3): ', np.array(outputAngles)[rankList[:3]]/math.pi*180.0)
            print('Target angles (top 3): ', np.array(targetAngles)[rankList[:3]]/math.pi*180.0)

        if N > 2:
            meanInconsistency = sum(inconsistencyList)/6
            if printing:
                print('Mean inconsistency: %.2f°' % (meanInconsistency/math.pi * 180.0))
        else:
            meanInconsistency = args.removeThreshold/180.0 * math.pi + 1

        if meanInconsistency > args.removeThreshold/180.0 * math.pi:
            if printing:
                print('')
            return []

        offset = 0.0
        for i in range(thresholdIndex+1):
            offset = i/(i+1)*offset + (outputAngles[rankList[i]]-targetAngles[rankList[i]])/(i+1)

        indices = np.array(rankList)[rankList[:thresholdIndex+1]]
        targets = np.array(targetAngles) + offset
        meanError = np.mean(np.abs(targets[indices] - np.array(targetAngles)[indices]))

        if printing:
            print('Mean error: %.2f°\n' % (meanError/math.pi * 180.0))

        return list(targets)
        
    def __getitem__(self, targetIndex):
        mainIndex = self.toMainIndexList[targetIndex]
        frameIndex = self.toFrameIndexList[mainIndex]
        sequenceIndex = self.toSequenceIndexList[mainIndex]

        imagePath = '%s/training/image_0%d/%04d/%06d.png' % (args.dataRoot, args.cameraNumber, sequenceIndex, frameIndex)
        fullImage = Image.open(imagePath).convert('RGB')

        boundingBox = self.boundingBoxes[mainIndex]
        bbImage = fullImage.crop(boundingBox)
        target = self.selfSupervisedTargets[targetIndex]
        
        if random.random() > 0.5:
            bbImage = mirror(bbImage)
            target = np.sign(target)*math.pi - target

        bbImage = self.transform(bbImage)         
        return bbImage, target

    def __len__(self):
        return len(self.toMainIndexList)

class ValDataset(Dataset):
    def __init__(self, dictionary, percentage, root, transform):
        super(ValDataset, self).__init__()

        self.root = root
        self.transform = transform

        labelDir = '%s/label_0%d' % (root, args.cameraNumber)
        sequenceIds = dictionary['indices']
        readMode = dictionary['readMode']
        sequenceLabelsDict = readLabelFiles(labelDir, sequenceIds, readMode=readMode)

        self.boundingBoxes = []
        self.targets = []

        self.toFrameIndexList = []
        self.toSequenceIndexList = []

        for sequenceIndex, objectLabels in sequenceLabelsDict.items():
            beginIndex = int(len(objectLabels)*(1.0-percentage))

            for objectLabel in objectLabels[beginIndex:]:
                if objectLabel['type'] == 'Car':
                    left = objectLabel['left']
                    top = objectLabel['top']
                    right = objectLabel['right']
                    bottom = objectLabel['bottom']   

                    self.boundingBoxes.append((left, top, right, bottom))
                    self.targets.append(objectLabel['alpha'])

                    self.toFrameIndexList.append(objectLabel['frame'])
                    self.toSequenceIndexList.append(sequenceIndex)
        
    def __getitem__(self, index):
        frameIndex = self.toFrameIndexList[index]
        sequenceIndex = self.toSequenceIndexList[index]

        imagePath = '%s/image_0%d/%04d/%06d.png' % (self.root, args.cameraNumber, sequenceIndex, frameIndex)
        fullImage = Image.open(imagePath).convert('RGB')

        boundingBox = self.boundingBoxes[index]
        bbImage = fullImage.crop(boundingBox)

        bbImage = self.transform(bbImage)
        target = self.targets[index]
        
        return bbImage, target

    def __len__(self):
        return len(self.toFrameIndexList)

class TestDataset(Dataset):
    def __init__(self, dictionary, root, transform):
        super(TestDataset, self).__init__()

        self.root = root
        self.transform = transform

        labelDir = '%s/label_0%d' % (root, args.cameraNumber)
        sequenceIds = dictionary['indices']
        readMode = dictionary['readMode']
        sequenceLabelsDict = readLabelFiles(labelDir, sequenceIds, readMode=readMode)

        self.boundingBoxes = []
        self.toFrameIndexList = []
        self.toSequenceIndexList = []

        for sequenceIndex, objectLabels in sequenceLabelsDict.items():
            for objectLabel in objectLabels:
                if objectLabel['type'] == 'Car':
                    left = objectLabel['left']
                    top = objectLabel['top']
                    right = objectLabel['right']
                    bottom = objectLabel['bottom']   

                    self.boundingBoxes.append((left, top, right, bottom))
                    self.toFrameIndexList.append(objectLabel['frame'])
                    self.toSequenceIndexList.append(sequenceIndex)
        
    def __getitem__(self, index):
        frameIndex = self.toFrameIndexList[index]
        sequenceIndex = self.toSequenceIndexList[index]

        imagePath = '%s/image_0%d/%04d/%06d.png' % (self.root, args.cameraNumber, sequenceIndex, frameIndex)
        fullImage = Image.open(imagePath).convert('RGB')

        boundingBox = self.boundingBoxes[index]
        bbImage = fullImage.crop(boundingBox)
        bbImage = self.transform(bbImage)
        
        return bbImage

    def __len__(self):
        return len(self.toFrameIndexList)

''' =================================================================================== '''
''' ------------------------- Collection of smaller functions ------------------------- '''
''' =================================================================================== '''

def getAnglePath():
    if args.loadModelPath:
        return args.loadModelPath
    elif args.loadModelRendering:
        return '%s/AngleEstimator/%s/%d.pt' % (args.virtualModelsRoot, args.loadModelRendering.capitalize(), args.loadModelNumber)
    elif args.loadModelNumber > 0:
        return '%s/Models/AngleEstimator/%s/%d.pt' % (args.experimentRoot, args.split, args.loadModelNumber)
    else:
        return ''

def getDevice():
    if torch.cuda.is_available():
        return 'cuda'
    else:
        raise RuntimeError('No GPU available')

def getOptimizer(model):
    if args.optimizer == 'SGD':
        from torch.optim import SGD
        optimizer = SGD(model.parameters(), lr=args.learningRate, momentum=args.optimizerMomentum, weight_decay=args.weightDecay)

    return optimizer

def getScheduler(optimizer):
    if args.scheduler == 'MultiStepLR':
        from torch.optim.lr_scheduler import MultiStepLR
        scheduler = MultiStepLR(optimizer, args.multiStepLrMilestones, gamma=args.stepLrGamma)

    return scheduler

def getTrainTransform():
    resize = transforms.Resize((224, 224))
    toTensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return transforms.Compose([resize, toTensor, normalize])

def getTrainValDict():
    if args.split == 'All':
        trainDict, valDict = getSplitAll()
    elif args.split == 'Complementary':
        trainDict, valDict = getSplitComplementary()
    elif args.split == 'Halves':
        trainDict, valDict = getSplitHalves()
    elif args.split == 'VirtualKitti':
        trainDict, valDict = getSplitVirtualKitti()

    if args.split == 'All':
        trainPercentage = args.splitPercentage
        valPercentage = 1.0-trainPercentage
    else:
        trainPercentage = 1.0
        valPercentage = 1.0

    return trainDict, trainPercentage, valDict, valPercentage

def getValTransform():
    resize = transforms.Resize((224, 224))
    toTensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return transforms.Compose([resize, toTensor, normalize])

def initialization():
    anglePath = getAnglePath()

    if args.representation == 'Single':
        angleModel = loadSingleAngleModel(anglePath)
    elif args.representation == 'Double':
        angleModel = loadDoubleAngleModel(anglePath)

    optimizer = getOptimizer(angleModel)
    scheduler = getScheduler(optimizer)
    startCycle = 0
    startEpoch = 0

    if args.resume:
        angleModel, optimizer, scheduler, startCycle, startEpoch = loadCheckpoint(angleModel, optimizer, scheduler)

    return angleModel, optimizer, scheduler, startCycle, startEpoch

def loadCheckpoint(model, optimizer, scheduler):
    loadPath = '%s/Checkpoints/SelfSupervisedAngle/%s/%d.pt' % (args.experimentRoot, args.split, args.experimentNumber)

    if os.path.isfile(loadPath):
        print('Loading checkpoint from experiment %d ...' % (args.experimentNumber))
        checkpoint = torch.load(loadPath)

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        startCycle = checkpoint['cycle']
        startEpoch = checkpoint['epoch']
        print('Checkpoint loaded: Cycle %d | Epoch %d\n' % (startCycle+1, startEpoch+1))

    return model, optimizer, scheduler, startCycle, startEpoch

def loadDataset(angleModel, device):
    trainDict, trainPercentage, valDict, valPercentage = getTrainValDict()
    trainRoot, valRoot = ['%s/training' % (args.dataRoot)] * 2

    trainTransform = getTrainTransform()
    valTransform = getValTransform()

    trainDataset = TrainDataset(angleModel, device, trainDict, trainPercentage, trainRoot, trainTransform)
    valDataset = ValDataset(valDict, valPercentage, valRoot, valTransform)

    trainLoader = DataLoader(trainDataset, batch_size=args.batchSize, shuffle=True, num_workers=args.workers, pin_memory=True)
    valLoader = DataLoader(valDataset, batch_size=args.batchSize, shuffle=False, num_workers=args.workers, pin_memory=True)

    trainSize = len(trainDataset)
    valSize = len(valDataset)

    print('Train size: %d' % (trainSize))
    print('Val size: %d\n' % (valSize))

    return trainLoader, trainSize, valLoader, valSize

def loadDoubleAngleModel(anglePath):
    backbone = resnext50_32x4d(pretrained=True)

    numberOfFeatures = backbone.fc.in_features
    backbone.fc = nn.Linear(numberOfFeatures, 2)
    
    angleModel = nn.Sequential(backbone, Normalize(), SinCosToAngle())

    if anglePath:
        print('Loading angle model ...')
        savedModel = torch.load(anglePath)
        angleModel.load_state_dict(savedModel['model'])
        print('Model loaded.\n')

    return angleModel

def loadSingleAngleModel(anglePath):
    angleModel = resnext50_32x4d(pretrained=True)
    numberOfFeatures = angleModel.fc.in_features
    angleModel.fc = nn.Linear(numberOfFeatures, 1)

    if anglePath:
        print('Loading angle model ...')
        savedModel = torch.load(anglePath)
        angleModel.load_state_dict(savedModel['model'])
        print('Model loaded.\n')

    return angleModel

def printEvaluation():
    print('--------------------------')
    print('|       Evaluation       |')
    print('--------------------------')
    print('')

def printInitialization():
    print('--------------------------')
    print('|     Initialization     |')
    print('--------------------------')
    print('')

def printParameters():
    print('')
    print('*******************************')
    print('*       Angle estimator       *')
    print('*******************************')
    print('')

    print('--------------------------')
    print('|    Hyper-parameters    |')
    print('--------------------------')
    print('')

    print('Batch size:        ' + str(args.batchSize))
    print('Camera number:     ' + str(args.cameraNumber))
    print('Huber Quad. size:  ' + str(args.huberQuadSize))
    print('Label number:      ' + str(args.labelNumber))
    print('Learning rate:     ' + str(args.learningRate))

    if args.loadModelNumber > 0:
        print('Load model number: ' + str(args.loadModelNumber))

    if args.loadModelRendering:
        print('Load rendering:    ' + str(args.loadModelRendering))

    if args.loadModelPath:
        print('Load model path:   ' + str(args.loadModelPath))

    print('Milestones:        ' + str(args.multiStepLrMilestones))
    print('Min bb. size:      ' + str(args.minBbSize))
    print('Optimizer:         ' + str(args.optimizer))
    print('Optim. momentum:   ' + str(args.optimizerMomentum))
    print('Prune threshold:   ' + str(args.pruneThreshold))
    print('Remove threshold:  ' + str(args.removeThreshold))
    print('Representation:    ' + str(args.representation))
    print('Resume:            ' + str(args.resume))
    print('Scheduler:         ' + str(args.scheduler))
    print('Split:             ' + str(args.split))

    if args.split == 'All':
        print('Split percentage:  ' + str(args.splitPercentage))

    print('Step LR gamma:     ' + str(args.stepLrGamma))
    print('Training cycles:   ' + str(args.trainingCycles))
    print('Training epochs:   ' + str(args.trainingEpochs))
    print('Weight decay:      ' + str(args.weightDecay))
    print('Workers:           ' + str(args.workers))
    print('')

def printProgress(cycle, epoch):
    epochString = 'Epoch %d/%d' % (epoch+1, args.trainingEpochs)

    if args.trainingCycles == 1:
        print(epochString)
        print('-' * (len(epochString)))

    elif args.trainingCycles > 1:
        cycleString = 'Cycle %d/%d' % (cycle+1, args.trainingCycles)
        progressString = '%s | %s' % (cycleString, epochString) 
        print(progressString)
        print('-' * (len(progressString)))

def printTraining():
    print('--------------------------')
    print('|        Training        |')
    print('--------------------------')
    print('')

def removeCheckpoint():
    checkpointPath = '%s/Checkpoints/SelfSupervisedAngle/%s/%d.pt' % (args.experimentRoot, args.split, args.experimentNumber)
    os.remove(checkpointPath)

def saveCheckpoint(model, optimizer, scheduler, cycle, epoch):
    checkpointDir = '%s/Checkpoints/SelfSupervisedAngle/%s' % (args.experimentRoot, args.split)
    os.makedirs(checkpointDir, exist_ok=True)

    savePath = '%s/%d.pt' % (checkpointDir, args.experimentNumber)
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'cycle': cycle, 'epoch': epoch}, savePath)

def saveModel(model):
    modelDir = '%s/Models/SelfSupervisedAngle/%s' % (args.experimentRoot, args.split)
    os.makedirs(modelDir, exist_ok=True)

    savePath = '%s/%d.pt' % (modelDir, args.experimentNumber)
    torch.save({'model': model.state_dict()}, savePath)

def train(device, trainLoader, trainSize, angleModel, optimizer):
    criterion = nn.SmoothL1Loss(reduction='sum')
    scaleFactor = 0.5*args.huberQuadSize * math.pi/180.0
        
    angleModel.train()
    trainLoss = 0.0

    startTime = time.time()

    for bbImages, targets in trainLoader:
        optimizer.zero_grad()

        bbImages = bbImages.to(device)
        targets = targets.to(device, dtype=torch.float)

        predictions = angleModel(bbImages)
        predictions = predictions.squeeze()

        loss = criterion(predictions/scaleFactor, targets/scaleFactor)
        loss.backward()

        optimizer.step()
        trainLoss += loss.item()

    endTime = time.time()

    print('Train loss: %.3f' % (trainLoss/trainSize))
    print('Train time: %.0f min %.0f s\n' % ((endTime-startTime)//60, (endTime-startTime)%60))    

def validate(device, valLoader, valSize, angleModel):
    errorCriterion = nn.L1Loss(reduction='none')
    lossCriterion = nn.SmoothL1Loss(reduction='sum')
    scaleFactor = 0.5*args.huberQuadSize * math.pi/180.0

    angleModel.eval()
    valErrorList = []
    valLoss = 0.0
    
    startTime = time.time()

    with torch.set_grad_enabled(False):
        for bbImages, targets in valLoader:
            bbImages = bbImages.to(device)
            targets = targets.to(device, dtype=torch.float)

            predictions = angleModel(bbImages)
            predictions = predictions.squeeze()

            error = errorCriterion(predictions, targets)
            loss = lossCriterion(predictions/scaleFactor, targets/scaleFactor)

            valErrorList.extend(error.tolist())
            valLoss += loss.item()

    endTime = time.time()

    meanValError = np.mean(np.array(valErrorList))
    medianValError = np.median(np.array(valErrorList))
    
    print('Val error (mean): %.3f°' % (meanValError * 180.0/math.pi))
    print('Val error (median): %.3f°' % (medianValError * 180.0/math.pi))
    print('Val loss: %.3f' % (valLoss/valSize))
    print('Val time: %.0f min %.0f s\n' % ((endTime-startTime)//60, (endTime-startTime)%60))

''' ================================================================= '''
''' ------------------------- Main function ------------------------- '''
''' ================================================================= '''

def main():

    ''' ================================================================== '''
    ''' ------------------------- Initialization ------------------------- '''
    ''' ================================================================== '''

    global args
    args = argparser.parse_args()
    printParameters()

    printInitialization()
    device = getDevice()
    
    angleModel, optimizer, scheduler, startCycle, startEpoch = initialization()
    angleModel.to(device)

    trainLoader, trainSize, valLoader, valSize = loadDataset(angleModel, device)

    printEvaluation()
    validate(device, valLoader, valSize, angleModel)
    printTraining()

    ''' ============================================================ '''
    ''' ------------------------- Training ------------------------- '''
    ''' ============================================================ '''    

    for cycle in range(startCycle, args.trainingCycles):
        if cycle >= 1:
            trainLoader.dataset.getSelfSupervisedTargets(angleModel, device)
            optimizer = getOptimizer(angleModel)
            scheduler = getScheduler(optimizer)

        for epoch in range(startEpoch, args.trainingEpochs):
            saveCheckpoint(angleModel, optimizer, scheduler, cycle, epoch)
            printProgress(cycle, epoch)

            train(device, trainLoader, trainSize, angleModel, optimizer)
            validate(device, valLoader, valSize, angleModel)
            scheduler.step()

    ''' ================================================================================= '''
    ''' ------------------------- Save final model and clean-up ------------------------- '''
    ''' ================================================================================= '''

    saveModel(angleModel)
    removeCheckpoint()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass


