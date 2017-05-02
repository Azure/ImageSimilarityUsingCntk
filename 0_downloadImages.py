# -*- coding: utf-8 -*-
from helpers import *
locals().update(importlib.import_module("PARAMETERS").__dict__)


####################################
# Parameter
####################################
downloadTopNImages = sys.maxsize  #set to e.g. 50 to only download the first 50 of the 428 images


####################################
# Main
####################################
makeDirectory(rootDir + "/data/")
makeDirectory(imgDir)

counter = 0
imgUrls = readTable(imgUrlsPath)
imgUrls = randomizeList(imgUrls)
for index, (label, url) in enumerate(imgUrls):
    # Skip image if was already downloaded
    outImgPath = os.path.join(imgDir, label, str(index) + ".jpg")
    if os.path.exists(outImgPath):
        continue

    # Download image
    print("Downloading image {} of {}: label={}, url={}".format(index, len(imgUrls), label, url))
    data = downloadFromUrl(url, False)
    if len(data) > 0:
        makeDirectory(os.path.join(imgDir, label))
        writeBinaryFile(outImgPath, data)

        # Sanity check: delete image if it is corrupted
        try:
            imread(outImgPath)
            counter += 1
        except:
            print("Removing corrupted image {}, url={}".format(outImgPath, url))
            os.remove(outImgPath)
print("Successfully downloaded {} of the {} image urls.".format(counter, len(imgUrls)))
print("DONE.")
