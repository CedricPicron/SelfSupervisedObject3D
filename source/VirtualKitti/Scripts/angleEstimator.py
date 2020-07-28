import argparse
import math
import numpy as np
import os
import random
import subprocess
import time
import torch

from pandas import read_csv

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
    default='/esat/ruchba/cpicron/Datasets/VirtualKitti',
    help='Path to dataset root.')
argparser.add_argument(
    '--experimentNumber',
    metavar='N',
    type=int,
    default=0,
    help='Number corresponding to experiment.')
argparser.add_argument(
    '--experimentRoot',
    metavar='PATH',
    type=str,
    default='/esat/ruchba/cpicron/ObjectDetector3D/VirtualKitti',
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
    '--loadModelRendering',
    metavar='MODE',
    type=str,
    choices=['all', 'clone', 'fog', 'morning', 'overcast', 'rain', 'sunset'],
    default='clone',
    help='Render mode on which loaded model was trained (relative load).')
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
    '--rendering',
    metavar='MODE',
    type=str,
    choices=['all', 'clone', 'fog', 'morning', 'overcast', 'rain', 'sunset'],
    default='clone',
    help='Render mode of images.')
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
    default='1.3.1',
    help='Version of Virtual Kitti dataset.')
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
    def __init__(self, percentage, transform, worldIndices, maxOcclusion=3, minBbSize=0.0):
        super(TrainDataset, self).__init__()
        self.transform = transform      

        self.boundingBoxes = []
        self.targets = []
        
        self.toImageIndexList = []
        self.toWorldIndexList = []
        
        for worldIndex in worldIndices:
            labelFileName = '%s/vkitti_%s_motgt/%04d_clone.txt' % (args.dataRoot, args.version, worldIndex)
            labels = read_csv(labelFileName, sep=' ', index_col=False)
            trainRange = range(int(len(labels)*percentage))

            for labelNumber in trainRange:
                if labels['label'][labelNumber] == 'Car':
                    left = labels['l'][labelNumber]
                    top = labels['t'][labelNumber]
                    right = labels['r'][labelNumber]
                    bottom = labels['b'][labelNumber]
                    
                    occlusionCondition = labels['occluded'][labelNumber] <= maxOcclusion
                    sizeCondition = (right-left)*(bottom-top) >= minBbSize

                    if occlusionCondition and sizeCondition:
                        self.boundingBoxes.append((left, top, right, bottom))                        
                        self.targets.append(labels['alpha'][labelNumber])

                        self.toImageIndexList.append(labels['frame'][labelNumber])
                        self.toWorldIndexList.append(worldIndex)
        
    def __getitem__(self, index):
        imageIndex = self.toImageIndexList[index]
        rendering = getRendering()
        worldIndex = self.toWorldIndexList[index]

        imagePath = '%s/vkitti_%s_rgb/%04d/%s/%05d.png' % (args.dataRoot, args.version, worldIndex, rendering, imageIndex)
        fullImage = Image.open(imagePath).convert('RGB')

        boundingBox = self.boundingBoxes[index]
        bbImage = fullImage.crop(boundingBox)
        target = self.targets[index]
        
        if random.random() > 0.5:
            bbImage = mirror(bbImage)
            target = np.sign(target)*math.pi - target

        bbImage = self.transform(bbImage)         
        return bbImage, target

    def __len__(self):
        return len(self.toImageIndexList)

class ValDataset(Dataset):
    def __init__(self, percentage, transform, worldIndices):
        super(ValDataset, self).__init__()
        self.transform = transform      

        self.boundingBoxes = []
        self.targets = []
        
        self.toImageIndexList = []
        self.toWorldIndexList = []
        
        for worldIndex in worldIndices:
            labelFileName = '%s/vkitti_%s_motgt/%04d_clone.txt' % (args.dataRoot, args.version, worldIndex)
            labels = read_csv(labelFileName, sep=' ', index_col=False)
            valRange = range(int(len(labels)*(1.0-percentage)), len(labels))

            for labelNumber in valRange:
                if labels['label'][labelNumber] == 'Car':
                    left = labels['l'][labelNumber]
                    top = labels['t'][labelNumber]
                    right = labels['r'][labelNumber]
                    bottom = labels['b'][labelNumber]

                    self.boundingBoxes.append((left, top, right, bottom))                        
                    self.targets.append(labels['alpha'][labelNumber])

                    self.toImageIndexList.append(labels['frame'][labelNumber])
                    self.toWorldIndexList.append(worldIndex)
        
    def __getitem__(self, index):
        imageIndex = self.toImageIndexList[index]
        rendering = getRendering()
        worldIndex = self.toWorldIndexList[index]

        imagePath = '%s/vkitti_%s_rgb/%04d/%s/%05d.png' % (args.dataRoot, args.version, worldIndex, rendering, imageIndex)
        fullImage = Image.open(imagePath).convert('RGB')

        boundingBox = self.boundingBoxes[index]
        bbImage = fullImage.crop(boundingBox)
        target = self.targets[index]

        bbImage = self.transform(bbImage)         
        return bbImage, target

    def __len__(self):
        return len(self.toImageIndexList)

class TestDataset(Dataset):
    def __init__(self, transform, worldIndices):
        super(TestDataset, self).__init__()
        self.transform = transform      

        self.boundingBoxes = []        
        self.toImageIndexList = []
        self.toWorldIndexList = []
        
        for worldIndex in worldIndices:
            labelFileName = '%s/vkitti_%s_motgt/%04d_clone.txt' % (args.dataRoot, args.version, worldIndex)
            labels = read_csv(labelFileName, sep=' ', index_col=False)

            for labelNumber in range(len(labels)):
                if labels['label'][labelNumber] == 'Car':
                    left = labels['l'][labelNumber]
                    top = labels['t'][labelNumber]
                    right = labels['r'][labelNumber]
                    bottom = labels['b'][labelNumber]

                    self.boundingBoxes.append((left, top, right, bottom))                      
                    self.toImageIndexList.append(labels['frame'][labelNumber])
                    self.toWorldIndexList.append(worldIndex)
        
    def __getitem__(self, index):
        imageIndex = self.toImageIndexList[index]
        rendering = getRendering()
        worldIndex = self.toWorldIndexList[index]

        imagePath = '%s/vkitti_%s_rgb/%04d/%s/%05d.png' % (args.dataRoot, args.version, worldIndex, rendering, imageIndex)
        fullImage = Image.open(imagePath).convert('RGB')

        boundingBox = self.boundingBoxes[index]
        bbImage = fullImage.crop(boundingBox)
        bbImage = self.transform(bbImage)

        return bbImage

    def __len__(self):
        return len(self.toImageIndexList)

''' =================================================================================== '''
''' ------------------------- Collection of smaller functions ------------------------- '''
''' =================================================================================== '''

def getAnglePath():
    if args.loadModelPath:
        return args.loadModelPath
    elif args.loadModelNumber > 0:
        return '%s/Models/AngleEstimator/%s/%d.pt' % (args.experimentRoot, args.loadModelRendering.capitalize(), args.loadModelNumber)
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

def getRendering():
    if args.rendering == 'all':
        renderingList = ['clone', 'fog', 'morning', 'overcast', 'rain', 'sunset']

        renderingIndex = random.randrange(len(renderingList))
        rendering = renderingList[renderingIndex]

    else:
        rendering = args.rendering

    return rendering

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
    loadPath = '%s/Checkpoints/AngleEstimator/%s/%d.pt' % (args.experimentRoot, args.rendering.capitalize(), args.experimentNumber)

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
    trainTransform = getTrainTransform()
    valTransform = getValTransform()

    trainWorldIndices = [1, 2, 6, 18, 20]
    valWorldIndices = [1, 2, 6, 18, 20]

    trainPercentage = 0.8
    valPercentage = 0.2

    trainDataset = TrainDataset(trainPercentage, trainTransform, trainWorldIndices, maxOcclusion=args.maxOcclusion, minBbSize=args.minBbSize)
    valDataset = ValDataset(valPercentage, valTransform, valWorldIndices)

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
        print('Load rendering:    ' + str(args.loadModelRendering))

    if args.loadModelPath:
        print('Load model path:   ' + str(args.loadModelPath))

    print('Max occlusion:     ' + str(args.maxOcclusion))
    print('Milestones:        ' + str(args.multiStepLrMilestones))
    print('Min bb. size:      ' + str(args.minBbSize))
    print('Optimizer:         ' + str(args.optimizer))
    print('Optim. momentum:   ' + str(args.optimizerMomentum))
    print('Rendering:         ' + str(args.rendering))
    print('Representation:    ' + str(args.representation))
    print('Resume:            ' + str(args.resume))
    print('Scheduler:         ' + str(args.scheduler))
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
    checkpointPath = '%s/Checkpoints/AngleEstimator/%s/%d.pt' % (args.experimentRoot, args.rendering.capitalize(), args.experimentNumber)
    subprocess.run(['rm', '-f', checkpointPath])

def saveCheckpoint(model, optimizer, scheduler, epoch):
    savePath = '%s/Checkpoints/AngleEstimator/%s/%d.pt' % (args.experimentRoot, args.rendering.capitalize(), args.experimentNumber)
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch}, savePath)

def saveModel(model):
    savePath = '%s/Models/AngleEstimator/%s/%d.pt' % (args.experimentRoot, args.rendering.capitalize(), args.experimentNumber)
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


