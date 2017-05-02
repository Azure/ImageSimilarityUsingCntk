# -*- coding: utf-8 -*-
from helpers import *
from helpers_cntk import *

# This script shows how to use the refined model to computes
# features for all images in a directory. Note that this
# script does not import the parameters and file paths
# specified in PARAMETERS.py.


####################################
# Parameters
####################################
imgDir    = "C:/Users/pabuehle/Desktop/newImgs/"          # Directory with images to featurize
modelPath = "C:/Users/pabuehle/Desktop/cntk.model"        # Path to trained CNTK model
outFeaturesPath = os.path.join(imgDir, "features.pickle") # Output location where computed features will be save to
boImgDirRecursive = True                                  # Set to "true" if images are in sub-folders within imgDir


####################################
# Main
####################################
random.seed(0)
printDeviceType()

# Load cntk model
print("Loading CNTK model from: " + modelPath)
if not os.path.exists(modelPath):
    raise Exception("Model file does not exist: " + modelPath)
model = load_model(modelPath)
node  = model.find_by_name("poolingLayer")
model = combine([node.owner])

# Get list of images and image IDs
if boImgDirRecursive:
    imgUUIs = []
    imgPaths = []
    subdirs = getDirectoriesInDirectory(imgDir)
    for subdir in subdirs:
        for filename in getFilesInDirectory(os.path.join(imgDir,subdir), ".jpg"):
            imgPaths.append(os.path.join(imgDir, subdir, filename))
            imgUUIs.append(subdir + "/" + filename)
else:
    filenames = getFilesInDirectory(imgDir, ".jpg")
    imgPaths = [os.path.join(imgDir, f) for f in filenames]
    imgUUIs  = filenames
if len(imgPaths) == 0:
    raise Exception("No jpeg images found in directory " + imgDir)

# Featurize each image
feats = dict()
width, height = find_by_name(model, "input").shape[1:]
print("CNTK model image input width = {} pixels and height = {} pixels.".format(width,height))
for index, (imgPath, imgUUI) in enumerate(zip(imgPaths,imgUUIs)):
    print("Processing image {} of {}: {}".format(index, len(imgPaths), imgPath))
    img = imread(imgPath)

    # Prepare DNN inputs
    # NOTE: CNTK rc1 (or higher) has a bug where during padding only the first dimension is assigned the pad value of 114.
    #       This bug can be simulated here by padColor = [114,0,0] instead of [114, 114, 114]
    imgPadded = imresizeAndPad(img, width, height, padColor = [114,0,0])
    arguments = {
        model.arguments[0]: [np.ascontiguousarray(np.array(imgPadded, dtype=np.float32).transpose(2, 0, 1))], # convert to CNTKs HWC format
    }

    # Run DNN model
    dnnOut = model.eval(arguments)
    feat = np.concatenate(dnnOut, axis=0).squeeze()
    feat = np.array(feat, np.float32)
    feats[imgUUI] = feat

print("Saving features to file {}.".format(outFeaturesPath))
saveToPickle(outFeaturesPath, feats)
print("DONE.")

