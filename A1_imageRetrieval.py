# -*- coding: utf-8 -*-
from helpers import *
from helpers_cntk import *
locals().update(importlib.import_module("PARAMETERS").__dict__)


####################################
# Parameters
####################################
queryImgPath = "./data/fashionTexture/striped/95.jpg"  #pointer to an image anywhere on disk
distMethod = 'weightedL2'  # Options: 'L2', 'weightedL2', 'weightedL2Prob', ...


####################################
# Main
####################################
random.seed(0)
printDeviceType()

#-----------------------------------------------------------------------------------------------------------------------
# Data loading
#-----------------------------------------------------------------------------------------------------------------------
# Read image
tstart = datetime.datetime.now()
queryImg = imread(queryImgPath)
print("Time loading image [ms]: {:.0f}".format((datetime.datetime.now() - tstart).total_seconds() * 1000))

# Load cntk model
tstart = datetime.datetime.now()
model = load_model(cntkRefinedModelPath)
node  = model.find_by_name("poolingLayer")
model = combine([node.owner])
print("Time loading DNN [ms]: {:.0f}".format((datetime.datetime.now() - tstart).total_seconds() * 1000))

# Load trained svm
tstart = datetime.datetime.now()
svmLearner    = loadFromPickle(svmPath)
svmBias    = svmLearner.base_estimator.intercept_
svmWeights = np.array(svmLearner.base_estimator.coef_[0])
print("Time loading SVM [ms]: {:.0f}".format((datetime.datetime.now() - tstart).total_seconds() * 1000))

# Load reference image features
tstart = datetime.datetime.now()
refImgInfos = loadFromPickle(imgInfosTestPath)
ImageInfo.allFeatures = loadFromPickle(featuresPath)
print("Time loading reference image features [ms]: {:.0f}".format((datetime.datetime.now() - tstart).total_seconds() * 1000))


#-----------------------------------------------------------------------------------------------------------------------
# Computation
#-----------------------------------------------------------------------------------------------------------------------
# Prepare DNN inputs
# NOTE: CNTK rc1 (or higher) has a bug where during padding only the first dimension is assigned the pad value of 114.
#       This bug can be simulated here by padColor = [114,0,0] instead of [114, 114, 114]
tstart = datetime.datetime.now()
imgPadded = imresizeAndPad(queryImg, rf_inputResoluton, rf_inputResoluton, padColor = [114,0,0])
arguments = {
    model.arguments[0]: [np.ascontiguousarray(np.array(imgPadded, dtype=np.float32).transpose(2, 0, 1))], # convert to CNTK's HWC format
}
print("Time cnkt input generation [ms]: {:.0f}".format((datetime.datetime.now() - tstart).total_seconds() * 1000))

# Run DNN model
tstart = datetime.datetime.now()
dnnOut = model.eval(arguments)
queryFeat = np.concatenate(dnnOut, axis=0).squeeze()
queryFeat = np.array(queryFeat, np.float32)
print("Time running DNN [ms]: {:.0f}".format((datetime.datetime.now() - tstart).total_seconds() * 1000))

# Compute distances between given query image and all other images
print("Distance computation using {} distance.".format(distMethod))
tstart = datetime.datetime.now()
dists = []
for refImgInfo in refImgInfos:
    refFeat = refImgInfo.getFeat()
    dist = computeVectorDistance(queryFeat, refFeat, distMethod, svm_boL2Normalize, svmWeights, svmBias, svmLearner)
    dists.append(dist)
print("Time computing {} pair-wise distances [ms]: {:.0f}".format(len(dists), (datetime.datetime.now() - tstart).total_seconds() * 1000))
print("DONE.")


#-----------------------------------------------------------------------------------------------------------------------
# Visualization
#-----------------------------------------------------------------------------------------------------------------------
pltAxes = [plt.subplot(4, 5, i + 1) for i in range(4*5)]
pltAxes[0].imshow(imconvertCv2Numpy(queryImg))
pltAxes[0].set_title("Query image")
[ax.axis('off') for ax in pltAxes]

sortOrder = np.argsort(dists)
if distMethod.lower().endswith('prob'):
    sortOrder = sortOrder[::-1]
sortOrder = sortOrder[:15]
for index, (ax, refIndex) in enumerate(zip(pltAxes[5:], sortOrder)):
    currDist = dists[refIndex]
    refImgPath = refImgInfos[refIndex].getImgPath(imgDir)
    ax.imshow(imconvertCv2Numpy(imread(refImgPath)))
    ax.set_title("{}: {:2.2f}".format(index,currDist))

plt.draw()
#plt.savefig("vis.jpg", dpi=200, bbox_inches='tight')
plt.show()
