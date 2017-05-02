# -*- coding: utf-8 -*-
from helpers import *
locals().update(importlib.import_module("PARAMETERS").__dict__)


####################################
# Main
####################################
random.seed(0)
makeDirectory(procDir)
imgFilenamesTest  = dict()
imgFilenamesTrain = dict()

print("Split images into train or test...")
subdirs = getDirectoriesInDirectory(imgDir)
for subdir in subdirs:
    filenames = getFilesInDirectory(imgDir + subdir, ".jpg")

    # Randomly assign images into train or test
    if imagesSplitBy == 'filename':
        filenames  = randomizeList(filenames)
        splitIndex = int(ratioTrainTest * len(filenames))
        imgFilenamesTrain[subdir] = filenames[:splitIndex]
        imgFilenamesTest[subdir]  = filenames[splitIndex:]

    # Randomly assign whole subdirectories to train or test
    elif imagesSplitBy == 'subdir':
        if random.random() < ratioTrainTest:
            imgFilenamesTrain[subdir] = filenames
        else:
            imgFilenamesTest[subdir]  = filenames
    else:
        raise Exception("Variable 'imagesSplitBy' has to be either 'filename' or 'subdir'")

    # Debug print
    if subdir in imgFilenamesTrain:
        print("Training: {:5} images in directory {}".format(len(imgFilenamesTrain[subdir]), subdir))
    if subdir in imgFilenamesTest:
        print("Testing:  {:5} images in directory {}".format(len(imgFilenamesTest[subdir]), subdir))

# Save assignments of images to train or test
saveToPickle(imgFilenamesTrainPath, imgFilenamesTrain)
saveToPickle(imgFilenamesTestPath,  imgFilenamesTest)

# Mappings label <-> id
lutId2Label = dict()
lutLabel2Id = dict()
for index, key in enumerate(imgFilenamesTrain.keys()):
    lutLabel2Id[key] = index
    lutId2Label[index] = key
saveToPickle(lutLabel2IdPath, lutLabel2Id)
saveToPickle(lutId2LabelPath, lutId2Label)

# Compute positive and negative image pairs
print("Generate training data ...")
imgInfosTrain = getImagePairs(imgFilenamesTrain, train_maxQueryImgsPerSubdir, train_maxNegImgsPerQueryImg)
saveToPickle(imgInfosTrainPath, imgInfosTrain)
print("Generate test data ...")
imgInfosTest = getImagePairs(imgFilenamesTest, test_maxQueryImgsPerSubdir, test_maxNegImgsPerQueryImg)
saveToPickle(imgInfosTestPath, imgInfosTest)

# Sanity check - make sure the test and training set have no images in common
if True:
    print("Verifying if training and test set are disjoint:")
    pathsTest  = getImgPaths(loadFromPickle(imgInfosTestPath))
    pathsTrain = getImgPaths(loadFromPickle(imgInfosTrainPath))

    # Make sure the training set and test set have zero overlap
    overlap = len(pathsTrain.intersection(pathsTest))
    if overlap == 0:
        print("   Check passed: Training and test set share no images.")
    else:
        raise Exception("Training and test set share %d images." % overlap)
print("DONE.")
