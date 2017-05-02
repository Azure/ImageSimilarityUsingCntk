# -*- coding: utf-8 -*-
import sys, os, importlib, pdb, random, datetime, collections, pickle, cv2, requests
import matplotlib.pyplot as plt, numpy as np, scipy.spatial.distance
from sklearn import svm, metrics, calibration
from PIL import Image, ExifTags

random.seed(0)


################################
# ImageInfo class and helpers
################################
class ImageInfo(object):
    allFeatures = []

    def __init__(self, fname, subdir, parent = None):
        self.fname  = fname
        self.subdir = subdir
        self.children = []
        self.parent = parent
        if parent:
            self.parent = self.shallowCopy(parent)

    def getFeat(self):
        if self.allFeatures == []:
            raise Exception("Need to set/load DNN features first using e.g. this line 'ImageInfo.allFeatures = loadFromPickle(featuresPath)'")
        key  = self.subdir + "/" + self.fname
        feat = np.array(self.allFeatures[key], np.float32)
        assert (len(feat) == 4096 or len(feat) == 2048 or len(feat) == 512 or len(feat) == 25088)
        return feat

    def getImg(self, rootDir):
        imgPath = self.getImgPath(rootDir)
        return imread(imgPath)

    def getImgPath(self, rootDir):
        return rootDir + self.subdir + "/" + self.fname

    def addChild(self, node):
        node.parent = self
        self.children.append(node)

    def isSameClassAsParent(self):
        return self.subdir == self.parent.subdir

    def shallowCopy(self, node):
        return ImageInfo(node.fname, node.subdir, node.parent)

    def display(self):
        print("Parent: " + self.node2Str(self))
        for childIndex,child in enumerate(self.children):
            print("   Child {:4} : {}".format(childIndex, self.node2Str(child)))

    def node2Str(self, node):
        return("fname = {}, subdir={}".format(node.fname, node.subdir)) #, node.parent)


def getImgPaths(imgInfos, rootDir=""):
    paths = set()
    for imgInfo in imgInfos:
        paths.add(rootDir + "/" + imgInfo.subdir + "/" + imgInfo.fname)
        for child in imgInfo.children:
            paths.add(rootDir + "/" + child.subdir + "/" + child.fname)
    return paths


def getRandomImgInfo(imgFilenames, subdirToExclude = None):
    subdirs = list(imgFilenames.keys())
    subdir  = getRandomListElement(subdirs)
    while subdir == subdirToExclude:
        subdir = getRandomListElement(subdirs)
    imgFilename = getRandomListElement(imgFilenames[subdir])
    return ImageInfo(imgFilename, subdir)



################################
# helper functions - svm
################################
def getImgPairsFeatures(imgInfos, metric, boL2Normalize):
    feats = []
    labels = []
    for queryImgIndex, queryImgInfo in enumerate(imgInfos):
        queryFeat = queryImgInfo.getFeat()
        if boL2Normalize:
            queryFeat /= np.linalg.norm(queryFeat, 2)

        for refImgInfo in queryImgInfo.children:
            refFeat = refImgInfo.getFeat()
            if boL2Normalize:
                refFeat /= np.linalg.norm(refFeat, 2)

            # Evaluate difference between the two images
            featDiff = queryFeat - refFeat
            if metric.lower() == 'diff':
                feat = featDiff
            elif metric.lower() == 'l1':
                feat = abs(featDiff)
            elif metric.lower() == 'l2':
                feat = featDiff ** 2
            else:
                raise Exception("Unknown metric: " + metric)
            feats.append(np.float32(feat))
            labels.append(int(refImgInfo.isSameClassAsParent()))
    return feats, labels


def mineHardNegatives(learner, imgFilenames, nrAddPerIter, featureDifferenceMetric, boL2Normalize,
                      maxNrRounds, initialThreshold = 1):
    hardNegatives = []
    roundCounterHardNegFound = 0
    hardNegThreshold = initialThreshold

    # Hard negative mining by repeatedly selecting a pair of images and adding to the
    # training set if they are misclassified by at least a certain threshold.
    for roundCounter in range(maxNrRounds):
        roundCounterHardNegFound += 1
        if len(hardNegatives) >= nrAddPerIter:
            break

        # Reduce threshold if no hard negative found after 1000 rounds
        if roundCounterHardNegFound > 1000:
            hardNegThreshold /= 2.0
            roundCounterHardNegFound = 0
            print("   Hard negative mining sampling round {:6d}: found {:4d} number of hard negatives; reducing hard negative threshold to {:3.3f}.".format(
                  roundCounter, len(hardNegatives), hardNegThreshold))

        # Sample two images from different ground truth class
        ImageInfo1 = getRandomImgInfo(imgFilenames)
        ImageInfo2 = getRandomImgInfo(imgFilenames, ImageInfo1.subdir)
        ImageInfo1.addChild(ImageInfo2)

        # Evaluate svm
        featCandidate, labelCandidate = getImgPairsFeatures([ImageInfo1], featureDifferenceMetric, boL2Normalize)
        assert (len(labelCandidate) == 1 and labelCandidate[0] == 0 and ImageInfo1.subdir != ImageInfo2.subdir)
        score = learner.decision_function(featCandidate)

        # If confidence is sufficiently high then add to list of hard negatives
        if score > hardNegThreshold:
            hardNegatives.append(featCandidate[0])
            roundCounterHardNegFound = 0
    print("   Hard negatives found: {}, after {} sampling rounds".format(len(hardNegatives), roundCounter+1))
    return hardNegatives


def getSampleWeights(labels, negPosRatio = 1):
    indsNegatives = np.where(np.array(labels) == 0)[0]
    indsPositives = np.where(np.array(labels) != 0)[0]
    negWeight = float(negPosRatio) * len(indsPositives) / len(indsNegatives)
    weights = np.array([1.0] * len(labels))
    weights[indsNegatives] = negWeight
    assert (abs(sum(weights[indsNegatives]) - negPosRatio * sum(weights[indsPositives])) < 10 ** -3)
    return weights


def plotScoreVsProbability(learner, feats_train, feats_test):
    probsTest   = learner.predict_proba(feats_test)[:, 1]
    probsTrain  = learner.predict_proba(feats_train)[:, 1]
    scoresTest  = learner.base_estimator.decision_function(feats_test)
    scoresTrain = learner.base_estimator.decision_function(feats_train)
    plt.scatter(scoresTrain, probsTrain, c='r', label = 'train')
    plt.scatter(scoresTest,  probsTest,  c='b', label = 'test')
    plt.ylim([-0.02, 1.02])
    plt.xlabel('SVM score')
    plt.ylabel('Probability')
    plt.title('Calibrated SVM - training set (red), test set (blue)')
    return plt



################################
# helper functions - general
################################
def getImagePairs(imgFilenames, maxQueryImgsPerSubdir, maxNegImgsPerQueryImg):
    # Get sub-directories with at least two images in them
    querySubdirs = [s for s in imgFilenames.keys() if len(imgFilenames[s]) > 1]

    # Generate pos and neg pairs for each subdir
    imgInfos = []
    for querySubdir in querySubdirs:
        queryFilenames = randomizeList(imgFilenames[querySubdir])

        # Pick at most 'maxQueryImgsPerSubdir' query images at random
        for queryFilename in queryFilenames[:maxQueryImgsPerSubdir]:
            queryInfo = ImageInfo(queryFilename, querySubdir)

            # Add one positive example at random
            refFilename = getRandomListElement(list(set(queryFilenames) - set([queryFilename])))
            queryInfo.children.append(ImageInfo(refFilename, querySubdir, queryInfo))
            assert(refFilename != queryFilename)

            # Add multiple negative examples at random
            for _ in range(maxNegImgsPerQueryImg):
                refSubdir   = getRandomListElement(list(set(querySubdirs) - set([querySubdir])))
                refFilename = getRandomListElement(imgFilenames[refSubdir])
                queryInfo.children.append(ImageInfo(refFilename, refSubdir, queryInfo))
                assert(refSubdir != querySubdir)

            # Store
            queryInfo.children = randomizeList(queryInfo.children)
            imgInfos.append(queryInfo)
    print("Generated image pairs for {} query images, each with 1 positive image pair and {} negative image pairs.".format(len(imgInfos), maxNegImgsPerQueryImg))
    return imgInfos


def getImgLabelMap(imgFilenames, imgDir, lut = None):
    table = []
    for label in imgFilenames.keys():
        for imgFilename in imgFilenames[label]:
            imgPath = imgDir + "/" + str(label) + "/" + imgFilename
            if lut != None:
                table.append((imgPath, lut[label]))
            else:
                table.append((imgPath, label))
    return table


def balanceDatasetUsingDuplicates(data):
    duplicates = []
    counts = collections.Counter(getColumn(data,1))
    print("Before balancing of training set:")
    for item in counts.items():
        print("   Class {:3}: {:5} exmples".format(*item))

    # Get duplicates to balance dataset
    targetCount = max(getColumn(counts.items(), 1))
    while min(getColumn(counts.items(),1)) < targetCount:
        for imgPath, label in data:
            if counts[label] < targetCount:
                duplicates.append((imgPath, label))
                counts[label] += 1

    # Add duplicates to original dataset
    print("After balancing: all classes now have {} images; added {} duplicates to the {} original images.".format(targetCount, len(duplicates), len(data)))
    data += duplicates
    counts = collections.Counter(getColumn(data,1))
    assert(min(counts.values()) == max(counts.values()) == targetCount)
    return data


def printFeatLabelInfo(title, feats, labels, preString = "   "):
    print(title)
    print(preString + "Number of examples: {}".format(len(feats)))
    print(preString + "Number of positive examples: {}".format(sum(np.array(labels) == 1)))
    print(preString + "Number of negative examples: {}".format(sum(np.array(labels) == 0)))
    print(preString + "Dimension of each example: {}".format(len(feats[0])))


def sklearnAccuracy(learner, feats, gtLabels):
    estimatedLabels = learner.predict(feats)
    confusionMatrix = metrics.confusion_matrix(gtLabels, estimatedLabels)
    return accsConfusionMatrix(confusionMatrix)



####################################
# Subset of helper library
# used in image similarity tutorial
####################################
# Typical meaning of variable names -- Computer Vision:
#    pt                     = 2D point (column,row)
#    img                    = image
#    width,height (or w/h)  = image dimensions
#    bbox                   = bbox object (stores: left, top,right,bottom co-ordinates)
#    rect                   = rectangle (order: left, top, right, bottom)
#    angle                  = rotation angle in degree
#    scale                  = image up/downscaling factor

# Typical meaning of variable names -- general:
#    lines,strings = list of strings
#    line,string   = single string
#    xmlString     = string with xml tags
#    table         = 2D row/column matrix implemented using a list of lists
#    row,list1D    = single row in a table, i.e. single 1D-list
#    rowItem       = single item in a row
#    list1D        = list of items, not necessarily strings
#    item          = single item of a list1D
#    slotValue     = e.g. "terminator" in: play <movie> terminator </movie>
#    slotTag       = e.g. "<movie>" or "</movie>" in: play <movie> terminator </movie>
#    slotName      = e.g. "movie" in: play <movie> terminator </movie>
#    slot          = e.g. "<movie> terminator </movie>" in: play <movie> terminator </movie>

def readFile(inputFile):
    # Reading as binary, to avoid problems with end-of-text characters.
    # Note that readlines() does not remove the line ending characters
    with open(inputFile,'rb') as f:
        lines = f.readlines()
    for i,s in enumerate(lines):
        removeLineEndCharacters(s.decode('utf8'))
    return [removeLineEndCharacters(s.decode('utf8')) for s in lines];

def writeFile(outputFile, lines, header=None):
    with open(outputFile,'w') as f:
        if header != None:
            f.write("%s\n" % header)
        for line in lines:
            f.write("%s\n" % line)

def writeBinaryFile(outputFile, data):
    with open(outputFile,'wb') as f:
        bytes = f.write(data)
    return bytes

def readTable(inputFile, delimiter='\t'):
    lines = readFile(inputFile);
    return splitStrings(lines, delimiter)

def writeTable(outputFile, table, header=None):
    lines = tableToList1D(table)
    writeFile(outputFile, lines, header)

def loadFromPickle(inputFile):
    with open(inputFile, 'rb') as filePointer:
         data = pickle.load(filePointer)
    return data

def saveToPickle(outputFile, data):
    p = pickle.Pickler(open(outputFile,"wb"))
    p.fast = True
    p.dump(data)

def makeDirectory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def getFilesInDirectory(directory, postfix=""):
    if not os.path.exists(directory):
        return []
    fileNames = [s for s in os.listdir(directory) if not os.path.isdir(directory + "/" + s)]
    if not postfix or postfix == "":
        return fileNames
    else:
        return [s for s in fileNames if s.lower().endswith(postfix)]

def getDirectoriesInDirectory(directory):
    return [s for s in os.listdir(directory) if os.path.isdir(directory + "/" + s)]

def downloadFromUrl(url, boVerbose = True):
    data = []
    url = url.strip()
    try:
        r = requests.get(url)
        data = r.content
    except:
        if boVerbose:
            print('Error downloading url {0}'.format(url))
    if boVerbose and data == []: # and r.status_code != 200:
        print('Error {} downloading url {}'.format(r.status_code, url))
    return data

def removeLineEndCharacters(line):
    if line.endswith('\r\n'):
        return line[:-2]
    elif line.endswith('\n'):
        return line[:-1]
    else:
        return line

def splitString(string, delimiter='\t', columnsToKeepIndices=None):
    if string == None:
        return None
    items = string.split(delimiter)
    if columnsToKeepIndices != None:
        items = getColumns([items], columnsToKeepIndices)
        items = items[0]
    return items

def splitStrings(strings, delimiter, columnsToKeepIndices=None):
    table = [splitString(string, delimiter, columnsToKeepIndices) for string in strings]
    return table

def getColumn(table, columnIndex):
    column = []
    for row in table:
        column.append(row[columnIndex])
    return column

def tableToList1D(table, delimiter='\t'):
    return [delimiter.join([str(s) for s in row]) for row in table]

def ToIntegers(list1D):
    return [int(float(x)) for x in list1D]

def mergeDictionaries(dict1, dict2):
    tmp = dict1.copy()
    tmp.update(dict2)
    return tmp

def getRandomNumber(low, high):
    randomNumber = random.randint(low,high)
    return randomNumber

def randomizeList(listND, containsHeader=False):
    if containsHeader:
        header = listND[0]
        listND = listND[1:]
    random.shuffle(listND)
    if containsHeader:
        listND.insert(0, header)
    return listND

def getRandomListElement(listND, containsHeader=False):
    if containsHeader:
        index = getRandomNumber(1, len(listND) - 1)
    else:
        index = getRandomNumber(0, len(listND) - 1)
    return listND[index]

def accsConfusionMatrix(confMatrix):
    perClassAccs = [(1.0 * row[rowIndex] / sum(row)) for rowIndex,row in enumerate(confMatrix)]
    return perClassAccs

def computeVectorDistance(vec1, vec2, method, boL2Normalize, weights = [], bias = [], learner = []):
    # Pre-processing
    if boL2Normalize:
        vec1 = vec1 / np.linalg.norm(vec1, 2)
        vec2 = vec2 / np.linalg.norm(vec2, 2)
    assert (len(vec1) == len(vec2))

    # Distance computation
    vecDiff = vec1 - vec2
    method = method.lower()
    if method == 'random':
        dist = random.random()
    elif method == 'l1':
        dist = sum(abs(vecDiff))
    elif method == 'l2':
        dist = np.linalg.norm(vecDiff, 2)
    elif method == 'normalizedl2':
        a = vec1 / np.linalg.norm(vec1, 2)
        b = vec2 / np.linalg.norm(vec2, 2)
        dist = np.linalg.norm(a - b, 2)
    elif method == "cosine":
        dist = scipy.spatial.distance.cosine(vec1, vec2)
    elif method == "correlation":
        dist = scipy.spatial.distance.correlation(vec1, vec2)
    elif method == "chisquared":
        dist = chiSquared(vec1, vec2)
    elif method == "normalizedchisquared":
        a = vec1 / sum(vec1)
        b = vec2 / sum(vec2)
        dist = chiSquared(a, b)
    elif method == "hamming":
        dist = scipy.spatial.distance.hamming(vec1 > 0, vec2 > 0)
    elif method == "mahalanobis":
        #assumes covariance matric is provided, e..g. using: sampleCovMat = np.cov(np.transpose(np.array(feats)))
        dist = scipy.spatial.distance.mahalanobis(vec1, vec2, sampleCovMat)
    elif method == 'weightedl1':
        feat = np.float32(abs(vecDiff))
        dist = np.dot(weights, feat) + bias
        dist = -float(dist)
        # assert(abs(dist - learnerL1.decision_function([feat])) < 0.000001)
    elif method == 'weightedl2':
        feat = (vecDiff) ** 2
        dist = np.dot(weights, feat) + bias
        dist = -float(dist)
    elif method == 'weightedl2prob':
        feat = (vecDiff) ** 2
        dist = learner.predict_proba([feat])[0][1]
        dist = float(dist)

    # elif method == 'learnerscore':
    #     feat = (vecDiff) ** 2
    #     dist = learner.base_estimator.decision_function([feat])[0]
    #     dist = -float(dist)
    else:
        raise Exception("Distance method unknown: " + method)
    assert (not np.isnan(dist))
    return dist

def rotationFromExifTag(imgPath):
    TAGSinverted = {v: k for k, v in list(ExifTags.TAGS.items())}
    orientationExifId = TAGSinverted['Orientation']
    try:
        imageExifTags = Image.open(imgPath)._getexif()
    except:
        imageExifTags = None

    #rotate the image if orientation exif tag is present
    rotation = 0
    if imageExifTags != None and orientationExifId != None and orientationExifId in imageExifTags:
        orientation = imageExifTags[orientationExifId]
        if orientation == 1 or orientation == 0:
            rotation = 0 #no need to do anything
        elif orientation == 6:
            rotation = -90
        elif orientation == 8:
            rotation = 90
        else:
            raise Exception("ERROR: orientation = " + str(orientation) + " not_supported!")
    return rotation

def imread(imgPath, boThrowErrorIfExifRotationTagSet = True):
    if not os.path.exists(imgPath):
        raise Exception("ERROR: image path does not exist.")
    rotation = rotationFromExifTag(imgPath)
    if boThrowErrorIfExifRotationTagSet and rotation != 0:
        print("Error: exif roation tag set, image needs to be rotated by %d degrees." % rotation)
    img = cv2.imread(imgPath)
    if img is None:
        raise Exception("ERROR: cannot load image " + imgPath)
    if rotation != 0:
        img = imrotate(img, -90).copy()  # To avoid occassional error: "TypeError: Layout of the output array img is incompatible with cv::Mat"
    return img

def imWidth(input):
    return imWidthHeight(input)[0]

def imHeight(input):
    return imWidthHeight(input)[1]

def imWidthHeight(input):
    if type(input) is str: #or type(input) is unicode:
        width, height = Image.open(input).size # This does not load the full image
    else:
        width =  input.shape[1]
        height = input.shape[0]
    return width,height

def imconvertCv2Numpy(img):
    (b,g,r) = cv2.split(img)
    return cv2.merge([r,g,b])

def imconvertCv2Pil(img):
    cv2_im = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)
    return pil_im

def imconvertPil2Cv(pilImg):
    return imconvertPil2Numpy(pilImg)[:, :, ::-1]

def imconvertPil2Numpy(pilImg):
    rgb = pilImg.convert('RGB')
    return np.array(rgb).copy()

def imresize(img, scale, interpolation = cv2.INTER_LINEAR):
    return cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=interpolation)

def imresizeMaxDim(img, maxDim, boUpscale = False, interpolation = cv2.INTER_LINEAR):
    scale = 1.0 * maxDim / max(img.shape[:2])
    if scale < 1  or boUpscale:
        img = imresize(img, scale, interpolation)
    else:
        scale = 1.0
    return img, scale

def imresizeAndPad(img, width, height, padColor):
    # resize image
    imgWidth, imgHeight = imWidthHeight(img)
    scale = min(float(width) / float(imgWidth), float(height) / float(imgHeight))
    imgResized = imresize(img, scale) #, interpolation=cv2.INTER_NEAREST)
    resizedWidth, resizedHeight = imWidthHeight(imgResized)

    # pad image
    top  = int(max(0, np.round((height - resizedHeight) / 2)))
    left = int(max(0, np.round((width - resizedWidth) / 2)))
    bottom = height - top - resizedHeight
    right  = width - left - resizedWidth
    return cv2.copyMakeBorder(imgResized, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=padColor)

def imrotate(img, angle):
    imgPil = imconvertCv2Pil(img)
    imgPil = imgPil.rotate(angle, expand=True)
    return imconvertPil2Cv(imgPil)

def imshow(img, waitDuration=0, maxDim = None, windowName = 'img'):
    if isinstance(img, str): # Test if 'img' is a string
        img = cv2.imread(img)
    if maxDim is not None:
        scaleVal = 1.0 * maxDim / max(img.shape[:2])
        if scaleVal < 1:
            img = imresize(img, scaleVal)
    cv2.imshow(windowName, img)
    cv2.waitKey(waitDuration)