#!/usr/bin/env 
"""
Contains tasks and classes for loading/manipulating an image sequence.
"""

import scaffold
import cv2
import cv
from xml.dom import minidom
import re
import os
import numpy as np
import _regions


CONFIG_FILE = scaffold.registerParameter("configFile", "", 
"""The location of the configuration file for this image series.""")

# extracts the name of an image series
_imageSeriesName = re.compile('Series([0-9]+)')
# extracts a properties file for an image series
_propertiesFile = re.compile(r"Series[0-9]+_Properties\.xml")

class ParseConfig(scaffold.Task):
    """
    Parses a configuration file to extract information about an image sequence
    and dumps that information into data.attrs and an imagePaths array.

    (Currently parses a _Properties.xml file).
    """

    dependencies = []
    name = "Parse Configuration"

    def isComplete(self):
        # for now we'll just assume if theres already a list of image paths 
        # that we've already loaded config information.
        return self.context.hasNode("imagePaths")

    def run(self):
        configFile = self._param(CONFIG_FILE)
        xml = minidom.parse(configFile)
        dims = xml.getElementsByTagName('DimensionDescription')
        a = self.context.attrs

        # config info
        a.folder     = os.path.dirname(configFile)
        a.name       = xml.getElementsByTagName('Name')[0].firstChild.data
        a.seriesNum  = int(_imageSeriesName.match(a.name).group(1))
        a.length     = int(dims[2].attributes['NumberOfElements'].value)
        a.pixel      = np.array([float(dims[i].attributes['Voxel'].value) for i in [0, 1]])
        a.shape      = np.array([int(dims[i].attributes['NumberOfElements'].value) for i in [0, 1]])
        a.duration   = float(dims[2].attributes['Length'].value)
        a.timeString = xml.getElementsByTagName('StartTime')[0].firstChild.data
        a.startTime  = a.timeString.split(' ')[1]
        a.dt         = a.duration/a.length
        a.imageSize  = a.shape*a.pixel
        a.channel    = 0 # TODO: handle multiple channels?

        # image paths
        def imagePath(imageNum):
            relative = '{0}_t{1}_ch{2:02d}.tif'\
                       .format(a.name, imageNum, a.channel)
            return os.path.join(a.folder, relative)
        
        numDigits = len(str(a.length - 1))
        imageNums = (str(i).zfill(numDigits) for i in range(a.length))
        self.context.createArray("imagePaths", [imagePath(num) for num in imageNums])
        self.context.flush()

    def export(self):
        return dict(info=self.context.attrs, 
                    imagePaths=self.context.node("imagePaths"))

class LoadImages(scaffold.Task):
    """
    Takes the config information for an image sequence and loads the actual
    images into an ImageSeq inside of data.cache[OriginalImages].
    """

    dependencies = [ParseConfig]
    name = "Load Images"

    def run(self):
        # all we need to do is give the imagePaths to a LazyImageSeq, which
        # will load the images on demand.
        pass

    def export(self):
        paths = self._import(ParseConfig, "imagePaths")
        return dict(images=LazyImageSeq(paths))


class ComputeForegroundMasks(scaffold.TaskInterface):
    name = "Compute Foreground Masks"
    willExport = ['masks']

MASK_LOW_THRESH = scaffold.registerParameter("maskLowThresh", -10.0, #-1.3
"""The difference from the mean pixel value (in mean differences) below which a 
pixel will be marked as in the foreground.""")
MASK_HIGH_THRESH = scaffold.registerParameter("maskHighThresh", 3.0, #1.0
"""The difference from the mean pixel value (in mean differences) above which a 
pixel will be marked as in the foreground.""")

class ComputeDifferenceMasks(scaffold.Task):
    """
    Determine which pixels are in the foreground of the original images of a 
    sequence by determining how much the differ from the average value.
    """

    name = "Compute Foreground Masks"
    dependencies = [LoadImages]

    scaffold.implements(ComputeDifferenceMasks, ComputeForegroundMasks)

    def run(self):
        self._images = self._import(LoadImages, "images")
        self._imageSize = self._images[0].shape
        self._lowThresh = self._param(MASK_LOW_THRESH)
        self._highThresh = self._param(MASK_HIGH_THRESH)
        
        self._computeAverageDiff()
        self._computeMasks()
        self._computeDiffs()

    def export(self):
        return dict(masks=ImageSeq(self._masks),
                    diffs=ImageSeq(self._diffs))

    def _computeAverageDiff(self):
        """Compute the mean and mean difference of each pixel in an image seq.

        Pixels are in the foreground if, for a particular image, they are 
        sufficiently far from the average (measured in mean differences).

        Also cast all of the images to floating point values in order to avoid
        rounding errors.
        """
        # loop through all of the images and simultaneously compute the sum
        # and the sum of differences for each pixel
        average     = self._images[0].astype(np.float32)
        # add 0.01 to avoid a zero divide
        averageDiff = 0.01 + np.zeros(self._imageSize, dtype=np.float32)
        previous    = self._images[0].astype(np.float32)
        
        for image in self._images[1:]:
            image = image.astype(np.float32)

            average += image
            averageDiff += np.abs(image - previous)
            previous = image

        # scale the sums so that they represent averages.
        average /= len(self._images)
        averageDiff /= len(self._images) - 1

        self._average = average
        self._averageDiff = averageDiff

    def _makeThresholdImage(self, threshold):
        """ 
        Returns an image where all the pixels are threshold mean differences
        away from the average.
        """
        return self._averageDiff*threshold + self._average

    def _computeMasks(self):
        """
        Identify all of the pixels which are statistical outliers as part of 
        the foreground.
        """
        highThresh = self._makeThresholdImage(self._highThresh)
        lowThresh = self._makeThresholdImage(self._lowThresh)
        
        self._masks = [(image < lowThresh) | (image > highThresh)
                       for image in self._images]

    def _computeDiffs(self):
        """
        Compute the difference from the mean (in std. dev. units).
        """
        self._diffs = [(image - self._average)/self._averageDiff
                       for image in self._images]


FEATURE_RADIUS = scaffold.registerParameter("featureRadius", 5,
"""The radius (in pixels) of the feature we are trying to identify. Used in the 
tophat morphological procedure.""")

class RemoveBackground(scaffold.Task):

    name = "Remove Background"
    dependencies = [LoadImages]

    def run(self):
        images = self._import(LoadImages, "images")
        featureRadius = self._param(FEATURE_RADIUS)
        tophatKernel = makeCircularKernel(featureRadius)

        lessBackground = []
        for image in images:
            scaled = forceRange(image, 0, 255)
            tophat = cv2.morphologyEx(scaled, cv2.MORPH_TOPHAT, tophatKernel)
            lessBackground.append(tophat)
        self._images = lessBackground

    def export(self):
        return dict(images=self._images)


class Watershed(scaffold.Task):

    name = "Watershed"
    dependencies = [RemoveBackground]

    def run(self):
        images = self._import(RemoveBackground, "images")
        erodeKernel = makeCircularKernel(2)
        dilateKernel = makeCircularKernel(2)

        regions = []
        isolated = []
        thresholds = []
        for image in images:
            #threshold, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
            threshold, binary = cv2.threshold(image, 24, 255, cv2.THRESH_BINARY)
            thresholds.append(threshold)

            # definitely background = 1, unsure = 0
            # rest are individually numbered particles
            unsure = cv2.dilate(binary, dilateKernel)
            _, background = cv2.threshold(unsure, 1, 1, cv2.THRESH_BINARY_INV)

            eroded = cv2.erode(binary, erodeKernel)
            contours, _ = cv2.findContours(eroded, 
                                           cv2.RETR_LIST, 
                                           cv2.CHAIN_APPROX_SIMPLE)
            foreground = np.zeros_like(background, dtype=np.int32)
            #for particleId in range(len(contours)):
            #    color = particleId + 2 # background = 1
            #    cv2.drawContours(foreground, contours, particleId, color)
            cv2.drawContours(foreground, contours, -1, 2)

            markers = foreground + background
            cv2.watershed(toColor(image), markers)

            regions.append(markers)

            select = image.copy()
            select[markers != 2] = 0
            isolated.append(select)

        self._regions = regions
        self._isolated = isolated
        self.context.debug("Mean threshold = {0}", sum(thresholds)/len(thresholds))

    def export(self):
        return dict(regions=self._regions, isolated=self._isolated)


REGION_SEGMENTATION = scaffold.registerParameter("regionSegmentation", 150.0,
"""Insert description here.""")
REGION_THRESHOLD = scaffold.registerParameter("regionThreshold", 10,
"""The minimum threshold applied to the region-average image.""")

class MergeStatisticalRegions(scaffold.Task):

    name = "Merge Statistical Regions"
    dependencies = [RemoveBackground]
    #dependencies = [ComputeForegroundMasks]

    scaffold.implements(MergeStatisticalRegions, ComputeForegroundMasks, 
                        default=True)

    def run(self):
        images = self._import(RemoveBackground, "images")
        #images = [forceRange(image*(image > 0), 0, 255).astype(np.uint8)
        #          for image in self._import(ComputeForegroundMasks, "diffs")]
        segParam = self._param(REGION_SEGMENTATION)
        threshold = self._param(REGION_THRESHOLD)

        self._masks = []
        self._groupedByRegions = []

        for image in images[:24]:
            image = cv2.GaussianBlur(image, (0, 0), 1.0)
            numRegions, average, regions = \
                    _regions.mergeStatisticalRegions(image, segParam)
            self._masks.append(average > threshold)
            self._groupedByRegions.append(regions)
            print numRegions

    def export(self):
        return dict(masks=self._masks, 
                    groupedByRegions=self._groupedByRegions)


class _ImageSeqBase(object):
    """
    Abstract base class for all image sequences (essentially just provides a 
    nice wrapper around a list of images with a few extra methods).
    """
    
    def __init__(self, cache, numImages):
        """Don't actually call this (unless subclassing _ImageSeqBase).

        cache is a dict that maps from indices (0...length) to images that are
        actually available in memory.
        numImages is the maximum number of images in a sequence (not the number
        already loaded).
        """
        self._cache = cache
        self._numImages = numImages
    
    # -------------------------------------------------------------------------
    # These just pass a few magic python methods down to the list of images, 
    # making this class act just like a list.

    def __len__(self): 
        return self._numImages
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self.__getitem__(i) for i in xrange(*key.indices(len(self)))]
        
        if key < 0: key += len(self)
        if key >= len(self): raise IndexError(key)

        if key in self._cache:
            return self._cache[key]
        else:
            return self._loadImage(key)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    
    def __str__(self):
        return "ImageSeq: {0} images".format(len(self))

    # -------------------------------------------------------------------------

    def _loadImage(self, index):
        """Loads the image at the given index.

        Needs to be implemented by subclasses.
        """
        raise NotImplemented()
        
    def copy(self):
        """Create an exact copy of this image sequence.

        Needs to be implemented by subclasses.
        """
        cacheCopy = dict((i, image.copy()) 
                         for i, image in self._cache.iteritems())
        return ImageSeq(cacheCopy, len(self))
    
    def writeMovie(self, fileName, fps=20):
        isColor = len(self[0].shape) == 3
        writer = cv2.VideoWriter(fileName, 
                                 fps=fps,
                                 fourcc=cv2.cv.CV_FOURCC(*"PIM1"),
                                 frameSize=self[0].shape[0:2], 
                                 isColor=isColor)
        
        for image in self:
            if image.dtype == np.bool:
                image = image.astype(np.uint8)*255
            writer.write(image)
        

class ImageSeq(_ImageSeqBase):
    """
    A sequence of images, all loaded in memory as NumPy arrays.
    """

    def __init__(self, images):
        """Creates a new sequence from the given list of images."""
        _ImageSeqBase.__init__(self, dict(enumerate(images)), len(images))

    def copy(self):
        """Create an exact copy of this image sequence."""
        cacheCopy = dict((i, image.copy()) 
                         for i, image in self._cache.iteritems())
        return ImageSeq(cacheCopy, len(self))

    def _loadImage(self, index):
        assert False # all already loaded

class LazyImageSeq(_ImageSeqBase):
    """
    A sequence of images from a set of paths, only loaded into memory when we
    actually need them.
    """
    
    def __init__(self, paths, imageCache=None):
        """Creates a new lazy sequence from the given image paths.

        paths should be a list of strings pointing to images.
        imageCache is an optional dict mapping from indicies to already loaded
        images.
        """
        imageCache = dict() if imageCache is None else imageCache
        _ImageSeqBase.__init__(self, imageCache, len(paths))
        self._paths = paths
    
    def __str__(self):
        return "LazyImageSeq: {0}/{1} in memory".format(len(self._cache), len(self))
    
    def __repr__(self):
        return "LazyImageSeq({0})".format(repr(self.paths))
    
    def _loadImage(self, index):
        # TODO: Allow color images?
        image = cv2.imread(self._paths[index], 0) # 0 = grayscale
        self._cache[index] = image
        return image

    def copy(self):
        """Create an exact copy of this image sequence."""
        cacheCopy = dict((i, image.copy()) 
                         for i, image in self._cache.iteritems())
        return LazyImageSeq(self._paths, cacheCopy)

def copyBlank(image):
    """Returns a blank (black/0's) image of the same size and type."""
    return np.zeros_like(image)

def toColor(image):
    return cv2.cvtColor(image, cv.CV_GRAY2RGB)

def forceRange(image, min, max):
    """
    Linearly adjusts the image to ensure that the lowest pixel value is min and
    the largest is max.
    """
    imageMin, imageMax = image.min(), image.max()
    return (image - imageMin)*((max - min)/(imageMax - imageMin)) + min

def binaryToGray(image):
    return image.astype(np.uint8)*255

def makeCircularKernel(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))