# -*- coding: utf-8 -*-
from helpers import *
from helpers_cntk import *
locals().update(importlib.import_module("PARAMETERS").__dict__)


################################################
# MAIN
################################################
# Init
printDeviceType()
makeDirectory(workingDir)
model = load_model(cntkRefinedModelPath)

# Compute features for each image and write to disk
print("Featurizing test set..")
featuresTest  = featurizeImages(model, imgFilenamesTestPath,  imgDir, workingDir + "/featurizer_map.txt", "poolingLayer", run_mbsize)
print("Featurizing training set..")
featuresTrain = featurizeImages(model, imgFilenamesTrainPath, imgDir, workingDir + "/featurizer_map.txt", "poolingLayer", run_mbsize)
features = mergeDictionaries(featuresTrain, featuresTest)
for feat in list(features.values()):
    assert(len(feat) == rf_modelOutputDimension)

# Save features to file
print("Writting CNTK outputs to file %s ..." % featuresPath)
saveToPickle(featuresPath, features)
print("DONE.")
