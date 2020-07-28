import sys
sys.path.insert(1, '../../../datasets/Kitti/Tracking/utils/Scripts')

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
    default='../../../datasets/NuScenes',
    help='Path to dataset root.')
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
    '--maxOcclusion',
    metavar='N',
    type=int,
    choices=[0, 1, 2, 3],
    default=3,
    help='Max occlusion level of images used during training.')
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
    '--representation',
    metavar='REPR',
    type=str,
    choices=['Single', 'Double'],
    default='Single',
    help='Type of angle representation.')
argparser.add_argument(
    '--resume',
    action = 'store_true',
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
    choices=['All', 'Boston', 'Singapore'],
    default='All',
    help='Train-validation split.')
argparser.add_argument(
    '--splitPercentage',
    metavar='PERCENT',
    type=float,
    default=0.8,
    help='Percentage of scene belonging to training (All-split case only).')
argparser.add_argument(
    '--stepLrGamma',
    metavar='GAMMA',
    type=float,
    default=0.1,
    help='Step scheduler decay rate.')
argparser.add_argument(
    '--trainingEpochs',
    metavar='EPOCHS',
    type=int,
    default=30,
    help='Total number of training epochs.')
argparser.add_argument(
    '--version',
    metavar='VERSION',
    type=str,
    choices=['v1.0-mini', 'v1.0-trainval'],
    default='v1.0-trainval',
    help='Dataset version.')
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
    def __init__(self, dictionary, percentage, root, transform):
        super(TrainDataset, self).__init__()

        self.root = root
        self.transform = transform
      
        labelDir = '%s/training/label_0%d' % (root, args.cameraNumber)
        sceneIds = dictionary['indices']
        readMode = dictionary['readMode']
        sceneLabelsDict = readLabelFiles(labelDir, sceneIds, readMode=readMode)

        self.boundingBoxes = []
        self.targets = []

        self.toFrameIndexList = []
        self.toNuScenesPathDict = {}
        self.toSequenceIndexList = []

        for sceneIndex, objectLabels in sceneLabelsDict.items():
            kittiPath = '%s/training/image_0%d/%04d.txt' % (self.root, args.cameraNumber, sceneIndex)
            nuScenesPaths = []

            with open(kittiPath, 'r') as imageFile:
                lines = imageFile.read().splitlines()
                for line in lines:
                    nuScenesPaths.append(line.split(' ')[-1])
                
            self.toNuScenesPathDict[sceneIndex] = nuScenesPaths
            endIndex = int(len(objectLabels)*percentage)

            for objectLabel in objectLabels[:endIndex]:
                if objectLabel['type'] == 'car':
                    left = objectLabel['left']
                    top = objectLabel['top']
                    right = objectLabel['right']
                    bottom = objectLabel['bottom']   
                    
                    occlusionCondition = objectLabel['occluded'] <= args.maxOcclusion
                    sizeCondition = (right-left)*(bottom-top) >= args.minBbSize

                    if occlusionCondition and sizeCondition:
                        self.boundingBoxes.append((left, top, right, bottom))
                        self.targets.append(objectLabel['alpha'])

                        self.toFrameIndexList.append(objectLabel['frame'])
                        self.toSequenceIndexList.append(sceneIndex)
        
    def __getitem__(self, index):
        frameIndex = self.toFrameIndexList[index]
        sceneIndex = self.toSequenceIndexList[index]
        nuScenesPath = self.toNuScenesPathDict[sceneIndex][frameIndex]

        fullImagePath = '%s/%s' % (self.root, nuScenesPath)
        fullImage = Image.open(fullImagePath).convert('RGB')

        boundingBox = self.boundingBoxes[index]
        bbImage = fullImage.crop(boundingBox)
        target = self.targets[index]
        
        if random.random() > 0.5:
            bbImage = mirror(bbImage)
            target = np.sign(target)*math.pi - target

        bbImage = self.transform(bbImage)      
        return bbImage, target

    def __len__(self):
        return len(self.toFrameIndexList)

class ValDataset(Dataset):
    def __init__(self, dictionary, percentage, root, transform):
        super(ValDataset, self).__init__()

        self.root = root
        self.transform = transform

        labelDir = '%s/training/label_0%d' % (root, args.cameraNumber)
        sceneIds = dictionary['indices']
        readMode = dictionary['readMode']
        sceneLabelsDict = readLabelFiles(labelDir, sceneIds, readMode=readMode)

        self.boundingBoxes = []
        self.targets = []

        self.toFrameIndexList = []
        self.toNuScenesPathDict = {}
        self.toSequenceIndexList = []

        for sceneIndex, objectLabels in sceneLabelsDict.items():
            kittiPath = '%s/training/image_0%d/%04d.txt' % (self.root, args.cameraNumber, sceneIndex)
            nuScenesPaths = []

            with open(kittiPath, 'r') as imageFile:
                lines = imageFile.read().splitlines()
                for line in lines:
                    nuScenesPaths.append(line.split(' ')[-1])
                
            self.toNuScenesPathDict[sceneIndex] = nuScenesPaths
            beginIndex = int(len(objectLabels)*(1.0-percentage))

            for objectLabel in objectLabels[beginIndex:]:
                if objectLabel['type'] == 'car':
                    left = objectLabel['left']
                    top = objectLabel['top']
                    right = objectLabel['right']
                    bottom = objectLabel['bottom']   

                    self.boundingBoxes.append((left, top, right, bottom))
                    self.targets.append(objectLabel['alpha'])

                    self.toFrameIndexList.append(objectLabel['frame'])
                    self.toSequenceIndexList.append(sceneIndex)
        
    def __getitem__(self, index):
        frameIndex = self.toFrameIndexList[index]
        sceneIndex = self.toSequenceIndexList[index]
        nuScenesPath = self.toNuScenesPathDict[sceneIndex][frameIndex]

        fullImagePath = '%s/%s' % (self.root, nuScenesPath)
        fullImage = Image.open(fullImagePath).convert('RGB')

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

        labelDir = '%s/testing/label_0%d' % (root, args.cameraNumber)
        sceneIds = dictionary['indices']
        readMode = dictionary['readMode']
        sceneLabelsDict = readLabelFiles(labelDir, sceneIds, readMode=readMode)

        self.boundingBoxes = []
        self.toFrameIndexList = []
        self.toNuScenesPathDict = {}
        self.toSequenceIndexList = []

        for sceneIndex, objectLabels in sceneLabelsDict.items():
            kittiPath = '%s/testing/image_0%d/%04d.txt' % (self.root, args.cameraNumber, sceneIndex)
            nuScenesPaths = []

            with open(kittiPath, 'r') as imageFile:
                lines = imageFile.read().splitlines()
                for line in lines:
                    nuScenesPaths.append(line.split(' ')[-1])
                
            self.toNuScenesPathDict[sceneIndex] = nuScenesPaths

            for objectLabel in objectLabels:
                if objectLabel['type'] == 'car':
                    left = objectLabel['left']
                    top = objectLabel['top']
                    right = objectLabel['right']
                    bottom = objectLabel['bottom']   

                    self.boundingBoxes.append((left, top, right, bottom))
                    self.toFrameIndexList.append(objectLabel['frame'])
                    self.toSequenceIndexList.append(sceneIndex)
        
    def __getitem__(self, index):
        frameIndex = self.toFrameIndexList[index]
        sceneIndex = self.toSequenceIndexList[index]
        nuScenesPath = self.toNuScenesPathDict[sceneIndex][frameIndex]

        fullImagePath = '%s/%s' % (self.root, nuScenesPath)
        fullImage = Image.open(fullImagePath).convert('RGB')

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
    if args.version == 'v1.0-mini':
        indices = list(range(10))

    elif args.version == 'v1.0-trainval':
        if args.split == 'All':
            indices = list(range(751))

        elif args.split == 'Boston':
            indices = list(range(59, 93))
            indices.extend(list(range(121, 136)))
            indices.extend(list(range(154, 171)))
            indices.extend(list(range(172, 214)))
            indices.extend(list(range(225, 249)))
            indices.extend(list(range(253, 261)))
            indices.extend(list(range(304, 315)))
            indices.extend(list(range(354, 608)))
            indices.extend(list(range(622, 639)))
            indices.extend(list(range(657, 702)))
            
        elif args.split == 'Singapore':
            indices = list(range(0, 59))
            indices.extend(list(range(93, 121)))
            indices.extend(list(range(136, 154)))
            indices.extend(list(range(171, 172)))
            indices.extend(list(range(214, 225)))
            indices.extend(list(range(249, 253)))
            indices.extend(list(range(261, 304)))
            indices.extend(list(range(315, 354)))
            indices.extend(list(range(608, 622)))
            indices.extend(list(range(639, 657)))
            indices.extend(list(range(702, 751)))

    trainDict = {'indices': indices, 'readMode': 'Full'}
    valDict = {'indices': indices, 'readMode': 'Full'}

    trainPercentage = args.splitPercentage
    valPercentage = 1.0-trainPercentage

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
    startEpoch = 0

    if args.resume:
        angleModel, optimizer, scheduler, startEpoch = loadCheckpoint(angleModel, optimizer, scheduler)

    return angleModel, optimizer, scheduler, startEpoch

def loadCheckpoint(model, optimizer, scheduler):
    loadPath = '%s/Checkpoints/AngleEstimator/%s/%d.pt' % (args.experimentRoot, args.split, args.experimentNumber)

    if os.path.isfile(loadPath):
        print('Loading checkpoint from experiment %d ...' % (args.experimentNumber))
        checkpoint = torch.load(loadPath)

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        startEpoch = checkpoint['epoch']
        print('Checkpoint loaded: Start epoch %d\n' % (startEpoch+1))

    return model, optimizer, scheduler, startEpoch

def loadDataset():
    trainDict, trainPercentage, valDict, valPercentage = getTrainValDict()
    trainRoot, valRoot = ['%s/%s' % (args.dataRoot, args.version)] * 2

    trainTransform = getTrainTransform()
    valTransform = getValTransform()

    trainDataset = TrainDataset(trainDict, trainPercentage, trainRoot, trainTransform)
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

def printEpochProgress(epoch):
    print('Epoch %d/%d' % (epoch+1, args.trainingEpochs))
    print('-' * (len(str(epoch+1)) + 9))

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
    print('Learning rate:     ' + str(args.learningRate))

    if args.loadModelNumber > 0:
        print('Load model number: ' + str(args.loadModelNumber))

    if args.loadModelPath:
        print('Load model path:   ' + str(args.loadModelPath))

    print('Max occlusion:     ' + str(args.maxOcclusion))
    print('Milestones:        ' + str(args.multiStepLrMilestones))
    print('Min bb. size:      ' + str(args.minBbSize))
    print('Optimizer:         ' + str(args.optimizer))
    print('Optim. momentum:   ' + str(args.optimizerMomentum))
    print('Representation:    ' + str(args.representation))
    print('Resume:            ' + str(args.resume))
    print('Scheduler:         ' + str(args.scheduler))
    print('Split:             ' + str(args.split))

    if args.split == 'All':
        print('Split percentage:  ' + str(args.splitPercentage))

    print('Step LR gamma:     ' + str(args.stepLrGamma))
    print('Training epochs:   ' + str(args.trainingEpochs))
    print('Version:           ' + str(args.version))
    print('Weight decay:      ' + str(args.weightDecay))
    print('Workers:           ' + str(args.workers))
    print('')

def printTraining():
    print('--------------------------')
    print('|        Training        |')
    print('--------------------------')
    print('')

def removeCheckpoint():
    checkpointPath = '%s/Checkpoints/AngleEstimator/%s/%d.pt' % (args.experimentRoot, args.split, args.experimentNumber)
    os.remove(checkpointPath)

def saveCheckpoint(model, optimizer, scheduler, epoch):
    checkpointDir = '%s/Checkpoints/AngleEstimator/%s' % (args.experimentRoot, args.split)
    os.makedirs(checkpointDir, exist_ok=True)

    savePath = '%s/%d.pt' % (checkpointDir, args.experimentNumber)
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch}, savePath)

def saveModel(model):
    modelDir = '%s/Models/AngleEstimator/%s' % (args.experimentRoot, args.split)
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
        predictions = predictions.squeeze(1)

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
            predictions = predictions.squeeze(1)

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
    trainLoader, trainSize, valLoader, valSize = loadDataset()

    angleModel, optimizer, scheduler, startEpoch = initialization()
    angleModel.to(device)
    printTraining()

    ''' ============================================================ '''
    ''' ------------------------- Training ------------------------- '''
    ''' ============================================================ '''    

    for epoch in range(startEpoch, args.trainingEpochs):
        saveCheckpoint(angleModel, optimizer, scheduler, epoch)
        printEpochProgress(epoch)

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


