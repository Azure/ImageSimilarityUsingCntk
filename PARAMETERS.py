# -*- coding: utf-8 -*-
import sys, os


#######################################
#######################################
datasetName = "fashionTexture"           # Name of the image directory, e.g. /data/myFashion_texture/


###################
# Parameters
###################
# Train and test splits (script: 1_prepareData.py)
ratioTrainTest = 0.75                   # Percentage of images used for training of the DNN and the SVM
imagesSplitBy  = 'filename'             # Options: 'filename' or 'subdir'. If 'subdir' is used, then all images in a subdir are assigned fully to train or test
train_maxQueryImgsPerSubdir = 100       # Number of query images used to train the SVM
train_maxNegImgsPerQueryImg = 100       # Number of negative images per training query image
test_maxQueryImgsPerSubdir  = 20        # Number of query images used to evaluate the SVM
test_maxNegImgsPerQueryImg  = 100       # Number of negative images per test query image

# Model refinement parameters (script: 2_refineDNN.py)
rf_modelFilename = "ResNet_18.model"    # Pre-trained ImageNet model
rf_inputResoluton = 224                 # DNN image input width and height in pixels
rf_dropoutRate    = 0.5                 # Droputout rate
rf_mbSize         = 16                  # Minibatch size (reduce if running out of memory)
rf_maxEpochs      = 45                  # Number of training epochs. Set to 0 to skip DNN refinement
rf_maxTrainImages = float('inf')        # Naximum number of training images per epoch. Set to float('inf') to use all images
rf_lrPerMb        = [0.01] * 20 + [0.001] * 20 + [0.0001]  # Learning rate schedule
rf_momentumPerMb  = 0.9                 # Momentum during gradient descent
rf_l2RegWeight    = 0.0005              # L2 regularizer weight during gradient descent
rf_boFreezeWeights      = False         # Set to 'True' to freeze all but the very last layer. Otherwise the full network is refined
rf_boBalanceTrainingSet = False         # Set to 'True' to duplicate images such that all labels have the same number of images

# SVM training params (script: 4_trainSVM.py)
svm_CVal = 0.1                           # Slack penality parameter C for SVM training
svm_boL2Normalize = True                 # Normalize 512-floats vector to be of unit length before SVM training
svm_featureDifferenceMetric = 'l2'       # Use weighted L2 distance
svm_hardNegMining_nrIterations = 0       # Number of hard negative mining iterations. Set to 0 to deactivate
svm_hardNegMining_nrAddPerIter = 5000    # Maximum number of hard negatives to add at each mining iteration
svm_hardNegMinging_maxNrRoundsPerIter = 100000 # Maximum number of image pairs tested during one hard negative iteration
svm_probabilityCalibrationNegPosRatio = 10 # Negative-to-positive ratio for Platt smoothing

###################
# Fixed parameters
# (do not modify)
###################
print("PARAMETERS: datasetName = " + datasetName)

# Directories
rootDir      = os.path.dirname(os.path.realpath(sys.argv[0])).replace("\\","/") + "/"
imgDir       = rootDir + "data/"    + datasetName + "/"
resourcesDir = rootDir + "resources/"
procDir      = rootDir + "proc/"    + datasetName + "/"
resultsDir   = rootDir + "results/" + datasetName + "/"
workingDir   = rootDir + "tmp/"

# Files
imgUrlsPath             = resourcesDir + "fashionTextureUrls.tsv"
imgInfosTrainPath       = procDir + "imgInfosTrain.pickle"
imgInfosTestPath        = procDir + "imgInfosTest.pickle"
imgFilenamesTrainPath   = procDir + "imgFilenamesTrain.pickle"
imgFilenamesTestPath    = procDir + "imgFilenamesTest.pickle"
lutLabel2IdPath         = procDir + "lutLabel2Id.pickle"
lutId2LabelPath         = procDir + "lutId2Label.pickle"
cntkRefinedModelPath    = procDir + "cntk.model"
cntkTestMapPath         = workingDir + "test_map.txt"
cntkTrainMapPath        = workingDir + "train_map.txt"
cntkPretrainedModelPath = os.path.join(rootDir, "resources", "cntk", rf_modelFilename)
featuresPath            = procDir + "features.pickle"
svmPath                 = procDir + "svm.np"

# Dimension of the DNN output, for "ResNet_18.model" this is 512
if rf_modelFilename.lower()   == "resnet_18.model" or rf_modelFilename.lower() == "resnet_34.model":
    rf_modelOutputDimension = 512
elif rf_modelFilename.lower() == "resnet_50.model":
    rf_modelOutputDimension = 2048
else:
    raise Exception("Model featurization dimension not specified.")