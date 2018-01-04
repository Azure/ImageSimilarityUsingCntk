# -*- coding: utf-8 -*-
from helpers import *
locals().update(importlib.import_module("PARAMETERS").__dict__)


####################################
# Parameters
####################################
distMethods = ['random', 'L1', 'L2', 'weighted'+svm_featureDifferenceMetric]  #'cosine', 'correlation', 'chiSquared', 'normalizedChiSquared']

# No need to change below parameters
boVisualizeResults  = True
boEvalOnTrainingSet = False  # Set to 'False' to evaluate using test set; 'True' to instead eval on training set
visualizationDir = resultsDir + "/visualizations_weightedl2/"



####################################
# Main
####################################
random.seed(0)

# Load trained svm
learner    = loadFromPickle(svmPath)
svmBias    = learner.base_estimator.intercept_
svmWeights = np.array(learner.base_estimator.coef_[0])

# Load data
print("Loading data...")
ImageInfo.allFeatures = loadFromPickle(featuresPath)
if not boEvalOnTrainingSet:
    imgInfos = loadFromPickle(imgInfosTestPath)
else:
    print("WARNING: evaluating on training set.")
    imgInfos = loadFromPickle(imgInfosTrainPath)

# Compute distances between all image pairs
print("Computing pair-wise distances...")
allDists = { queryIndex:collections.defaultdict(list) for queryIndex in range(len(imgInfos)) }
for queryIndex, queryImgInfo in enumerate(imgInfos):
    queryFeat = queryImgInfo.getFeat()
    if queryIndex % 50 == 0:
        print("Computing distances for query image {} of {}: {}..".format(queryIndex, len(imgInfos), queryImgInfo.fname))

    # Loop over all reference images and compute distances
    for refImgInfo in queryImgInfo.children:
        refFeat = refImgInfo.getFeat()
        for distMethod in distMethods:
            dist = computeVectorDistance(queryFeat, refFeat, distMethod, svm_boL2Normalize, svmWeights, svmBias)
            allDists[queryIndex][distMethod].append(dist)

# Evaluate
for distMethod in distMethods:
    correctRanks = []
    for queryIndex, queryImgInfo in enumerate(imgInfos):
        sortOrder = np.argsort(allDists[queryIndex][distMethod])
        boCorrectMatches = [child.isSameClassAsParent() for child in queryImgInfo.children]
        boCorrectMatches = np.array(boCorrectMatches)[sortOrder]
        positiveRank = np.where(boCorrectMatches)[0][0] + 1
        correctRanks.append(positiveRank)
    medianRank = round(np.median(correctRanks))
    top1Acc = 100.0 * np.sum(np.array(correctRanks) == 1) / len(correctRanks)
    top5Acc = 100.0 * np.sum(np.array(correctRanks) <= 5) / len(correctRanks)
    print("Distance {:>10}: top1Acc = {:5.2f}%, top5Acc = {:5.2f}%, meanRank = {:5.2f}, medianRank = {:2.0f}".format(distMethod, top1Acc, top5Acc, np.mean(correctRanks), medianRank))

# Visualize
if boVisualizeResults:
    makeDirectory(resultsDir)
    makeDirectory(visualizationDir)
    print("Visualizing results: writing images to " +  visualizationDir)

    # Loop over all query images
    for queryIndex, queryImgInfo in enumerate(imgInfos):
        print("   Visualizing result for query image: " + imgDir + queryImgInfo.fname)
        dists = allDists[queryIndex]["weightedl2"]

        # Find match with minimum distance (rank 1) and correct match
        sortOrder = np.argsort(dists)
        minDistIndex = sortOrder[0]
        correctIndex = np.where([child.isSameClassAsParent() for child in queryImgInfo.children])[0][0]
        minDist      = dists[minDistIndex]
        correctDist  = dists[correctIndex]
        queryImg     = queryImgInfo.getImg(imgDir)
        minDistImg   = imgInfos[queryIndex].children[minDistIndex].getImg(imgDir)
        correctImg   = imgInfos[queryIndex].children[correctIndex].getImg(imgDir)
        minDistLabel = imgInfos[queryIndex].children[minDistIndex].subdir

        # Visualize
        if minDistLabel == queryImgInfo.subdir:
            plt.rcParams['figure.facecolor'] = 'green' #correct ranking result
        else:
            plt.rcParams['figure.facecolor'] = 'red'
        pltAxes = [plt.subplot(1, 3, i+1) for i in range(3)]
        for ax, img, title in zip(pltAxes, (queryImg,minDistImg,correctImg),
                              ('Query image', 'MinDist match \n (dist={:3.2f})'.format(minDist), 'Correct match \n (dist={:3.2f})'.format(correctDist))):
            ax.imshow(imconvertCv2Numpy(img))
            ax.axis('off')
            ax.set_title(title)
        plt.draw()
        #plt.savefig(visualizationDir + "/" + queryImgInfo.fname.replace('/','-'), dpi=200, bbox_inches='tight', facecolor=plt.rcParams['figure.facecolor'])
        plt.show()

print("DONE.")


