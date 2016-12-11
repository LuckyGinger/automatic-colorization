#!/usr/bin/env python2
import sys, os, tarfile, shutil
from PIL import Image
from multiprocessing import Process, Queue

PREPEND_NAME = ".ppimage."
SIZE = 224

# Extracts all the images to the /tmp directory, and returns a list of the
# filepaths for each one.
def untar(tarFileName):
        dirName = "/tmp/" + PREPEND_NAME + tarFileName + "/"
        tf = tarfile.open(name=tarFileName)

        if os.path.exists(dirName):
                shutil.rmtree(dirName)

        os.makedirs(dirName)
        tf.extractall(dirName)

        return (dirName, tf.getnames())

# Takes a file name, and new width/height.
# This takes a file name (including path) and combines scaling and cropping of
# the image to resize the image. The file is replaced by the transformation.
# Returns true if the size was good, false if the size was too small or the
# image could not be opened.
def preprocess(fileName):
        try:
                # The L converts to gray-scale.
                img = Image.open(fileName).convert('L')
        except IOError:
                return False  # if this catch isn't here then the thread dies and doesn't finish the other files.

        img.save(fileName, 'JPEG')
        return True

# Returns how many images were too small, and how many images there were total.
# A Tuple of: (small, good + small).
def loop_images(dirName, fileNames, outTar, size=SIZE):
        smallCount = 1.0
        totalCount = 1.0
        for fn in fileNames:
                totalCount += 1
                goodSize = preprocess(dirName + fn)
                if not goodSize:
                        smallCount += 1
                        os.remove(dirName + fn)

        tf = tarfile.open(outTar, "w")
        try:
                for x in os.listdir(dirName):
                        tf.add(dirName + x, arcname=x)
        finally:
                tf.close()
        return (smallCount, totalCount)

# Takes in a portion of the system arguments, from start (inclusive), to stop
# (exclusive). Will create a new tar with nicely cropped images for each
# argument :)
def run(start, stop):
    if len(sys.argv) > 2:
        outDirName = sys.argv[-1] + '/'
        if not os.path.exists(outDirName): # Check if directory exists.
            print "Error: The directory", outDirName, "doesn't exist."
            return

        allSmallCount = 0.0
        allCount = 0.0
        for i in xrange(start, stop):
            print "Converting", sys.argv[i]
            dirName, fileNames = untar(sys.argv[i])

            smallCount, totalCount = loop_images(dirName, fileNames, outDirName + sys.argv[i])
            allSmallCount += smallCount
            allCount += totalCount

            if os.path.exists(dirName):
                shutil.rmtree(dirName)

        print "And the total percent of images that were too small is:", allSmallCount / allCount
    else:
        print "Usage: ppimage [files.tar] [output_directory]"

if __name__ == "__main__":
    # manually split the data for 8 cores
    length  = len(sys.argv) - 1
    c_one   = int(length * 0.125)
    c_two   = int(length * 0.25)
    c_three = int(length * 0.375)
    c_four  = int(length * 0.5)
    c_five  = int(length * 0.625)
    c_six   = int(length * 0.75)
    c_seven = int(length * 0.875)
    c_eight = length

    p1 = Process(target=run, args=(1,       c_one   - 1))
    p2 = Process(target=run, args=(c_one,   c_two   - 1))
    p3 = Process(target=run, args=(c_two,   c_three - 1))
    p4 = Process(target=run, args=(c_three, c_four  - 1))
    p5 = Process(target=run, args=(c_four,  c_five  - 1))
    p6 = Process(target=run, args=(c_five,  c_six   - 1))
    p7 = Process(target=run, args=(c_six,   c_seven - 1))
    p8 = Process(target=run, args=(c_seven, c_eight))

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()
    p8.start()
