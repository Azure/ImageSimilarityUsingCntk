# -*- coding: utf-8 -*-
from helpers import *
locals().update(importlib.import_module("PARAMETERS").__dict__)


####################################
# Main
####################################
random.seed(0)
ImageInfo.allFeatures = loadFromPickle(featuresPath)  # Load all DNN image features into memory

# Load training data
print("Generate training data...")
imgInfosTrain = loadFromPickle(imgInfosTrainPath)
#imgInfosTrain = imgInfosTrain[::50]
feats_train, labels_train = getImgPairsFeatures(imgInfosTrain, svm_featureDifferenceMetric, svm_boL2Normalize)
printFeatLabelInfo("Statistics training data:", feats_train, labels_train)

# Load test data
print("Generate test data...")
imgInfosTest = loadFromPickle(imgInfosTestPath)
#imgInfosTest = imgInfosTest[::25]
feats_test, labels_test = getImgPairsFeatures(imgInfosTest, svm_featureDifferenceMetric, svm_boL2Normalize)
printFeatLabelInfo("Statistics test data:", feats_test, labels_test)

# Perform one or more iterations of SVM training
print("\nTraining...")
bestAcc = float('-inf')
for hardNegIter in range(svm_hardNegMining_nrIterations+1):
    # Mine hard negatives
    # Note: use this functionality carefully since this will often find and wrongly annotated data to the
    #       training set and hence degrade the classifier.
    if hardNegIter > 0:
        print("Hard negative mining - iteration %d" % hardNegIter)
        if hardNegIter == 1:
            imgFilenamesTrain = loadFromPickle(imgFilenamesTrainPath)
        hardNegatives = mineHardNegatives(learner, imgFilenamesTrain, svm_hardNegMining_nrAddPerIter,
                                          svm_featureDifferenceMetric, svm_boL2Normalize, svm_hardNegMinging_maxNrRoundsPerIter)
        feats_train  += hardNegatives
        labels_train += [0] * len(hardNegatives)
        printFeatLabelInfo("   Statistics training data:", feats_train, labels_train, preString="      ")

    # Train svm
    print("   Start SVM training...")
    tstart  = datetime.datetime.now()
    learner = svm.LinearSVC(C=svm_CVal, class_weight='balanced', verbose=0)
    learner.fit(feats_train, labels_train)
    print("   Training time [labels_train]: " + str((datetime.datetime.now() - tstart).total_seconds() * 1000))
    print("   Training accuracy    = {:3.2f}%".format(100 * np.mean(sklearnAccuracy(learner, feats_train, labels_train))))
    testAcc = np.mean(sklearnAccuracy(learner, feats_test,  labels_test))
    print("   Test accuracy        = {:3.2f}%".format(100 * np.mean(testAcc)))

    # Store best model. Note that this should use a separate validation set, and not the test set.
    if testAcc > bestAcc:
        print("   Updating best model.")
        bestAcc = testAcc
        bestLearner = learner

# Calibrate probability. Set weights to balance dataset as if 10 times more negatives than positives
del learner
print("Probability calibration positive to negative ration = {}".format(svm_probabilityCalibrationNegPosRatio))
sampleWeights = getSampleWeights(labels_test, negPosRatio = svm_probabilityCalibrationNegPosRatio)
bestLearner = calibration.CalibratedClassifierCV(bestLearner, method='sigmoid', cv = "prefit") #sigmoid, isotonic
bestLearner.fit(feats_test, labels_test, sample_weight = sampleWeights)
saveToPickle(svmPath, bestLearner)
print("Wrote svm to: " + svmPath + "\n")

# Plot score vs probability of trained classifier
print("DONE. Showing SVM score vs probability.")
plotScoreVsProbability(bestLearner, feats_train, feats_test).show()

