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