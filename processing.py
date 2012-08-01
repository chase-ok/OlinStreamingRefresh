"""
Tasks related to processing an image sequence to idenfity and track particles.
"""

import cv
import cv2
import numpy as np
import math
import tables as tb
import images as im
import scaffold
from scipy import stats
from trackpy import tracking

from matplotlib import pyplot as plt


LOW_THRESH = scaffold.registerParameter("maskLowThresh", -1.8, #-1.3
"""The difference from the mean pixel value (in mean differences) below which a 
pixel will be marked as in the foreground.""")
HIGH_THRESH = scaffold.registerParameter("maskHighThresh", 1.3, #1.0
"""The difference from the mean pixel value (in mean differences) above which a 
pixel will be marked as in the foreground.""")

class ComputeForegroundMasks(scaffold.Task):
    """
    Determine which pixels are in the foreground of the original images of a 
    sequence by determining how much the differ from the average value.
    """

    name = "Compute Foreground Masks"
    dependencies = [im.LoadImages]

    def run(self):
        self._images = self._import(im.LoadImages, "images")
        self._imageSize = self._images[0].shape
        self._lowThresh = self._param(LOW_THRESH)
        self._highThresh = self._param(HIGH_THRESH)
        
        self._computeAverageDiff()
        self._computeMasks()

    def export(self):
        return dict(masks=im.ImageSeq(self._masks))

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
        averageDiff = np.zeros(self._imageSize, dtype=np.float32)
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

FEATURE_RADIUS = scaffold.registerParameter("featureRadius", 5,
"""The radius (in pixels) of the feature we are trying to identify. Used in the 
tophat morphological procedure.""")

class RemoveBackground(scaffold.Task):

    name = "Remove Background"
    dependencies = [im.LoadImages]

    def run(self):
        images = self._import(im.LoadImages, "images")
        featureRadius = self._param(FEATURE_RADIUS)
        tophatKernel = _makeCircularKernel(featureRadius)

        lessBackground = []
        for image in images:
            scaled = im.forceRange(image, 0, 255)
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
        erodeKernel = _makeCircularKernel(2)
        dilateKernel = _makeCircularKernel(2)

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
            cv2.watershed(im.toColor(image), markers)

            regions.append(markers)

            select = image.copy()
            select[markers != 2] = 0
            isolated.append(select)

        self._regions = regions
        self._isolated = isolated
        self.context.debug("Mean threshold = {0}", sum(thresholds)/len(thresholds))

    def export(self):
        return dict(regions=self._regions, isolated=self._isolated)


DILATION_RADIUS = scaffold.registerParameter("dilationRadius", 3,
"""Amount (in pixels) to dilate each image.""")
MORPH_THRESHOLD = scaffold.registerParameter("morphThreshold", 0.2,
"""Threshold for pixel values (0-1) after morphological procedures.""")
BLUR_SIGMA = scaffold.registerParameter("blurSigma", 2.0,
"""The std. dev. (in pixels) for the gaussian blur.""")
EXP_THRESHOLD = scaffold.registerParameter("expThreshold", 0.0001,
"""Threshold for pixel values after final dilation.""")

class IdentifyParticles(scaffold.Task):
    """
    Identify particles using a combination of foreground masks and the 
    algorithm described at https://github.com/tacaswell/tracking
    """

    name = "Identify Particles in Images"
    dependencies = [im.LoadImages, ComputeForegroundMasks]

    def run(self):
        self._images = self._import(im.LoadImages, "images")
        self._masks = self._import(ComputeForegroundMasks, "masks")
        self._imageSize = self._images[0].shape

        self._featureRadius = self._param(FEATURE_RADIUS)
        self._dilationRadius = self._param(DILATION_RADIUS)
        self._morphThreshold = self._param(MORPH_THRESHOLD)
        self._blurSigma = self._param(BLUR_SIGMA)
        self._expThreshold = self._param(EXP_THRESHOLD)

        # apply the foreground masks to the original images
        #self._images = [i*m.astype(np.uint8) for i, m in zip(self._images, self._masks)]

        self._doMorph()
        self._findLocalMax()

    def export(self):
        return dict(images=im.ImageSeq(self._maxed))

    def _doMorph(self):
        """Apply a gaussian blur and then a tophat."""
        tophatKernel = _makeCircularKernel(self._featureRadius)

        morphed = []
        for image in self._images:
            scaled = im.forceRange(image, 0.0, 1.0) # scale pixel values to 0-1
            blurred = cv2.GaussianBlur(scaled, (0, 0), self._blurSigma)
            tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, tophatKernel)
            # maximize contrast by forcing the range to be 0-1 again
            morphed.append(im.forceRange(tophat, 0.0, 1.0))

        self._morphed = morphed

    def _findLocalMax(self):
        """Find the centers of particles by thresholding and dilating."""
        dilationKernel = _makeCircularKernel(self._dilationRadius)

        maxed = []
        for image in self._morphed:
            # set pixels below morph thresh to 0
            threshed = stats.threshold(image, self._morphThreshold, newval=0.0)
            dilated = cv2.dilate(threshed, dilationKernel)
            # expThreshold is so named because the original algorithm 
            # originally exponentiated and then thresholded, which is the same
            # as flipping the sign and exponentiating the threshold.
            binary = (dilated - threshed) >= self._expThreshold
            maxed.append(binary)
        self._maxed = maxed

def _makeCircularKernel(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))

# An HDF5 table for storing identified ellipses
class EllipseTable(tb.IsDescription):
    frame    = tb.UInt32Col(pos=0)
    position = tb.Float32Col(pos=1, shape=2)
    angle    = tb.Float32Col(pos=2)
    area     = tb.Float32Col(pos=3)
    axes     = tb.Float32Col(pos=4, shape=2)

ELLIPSE_MIN_AREA = scaffold.registerParameter("ellipseMinArea", 0.5,
"""The minimum area required to identify an ellipse.""")
ELLIPSE_MAX_AREA = scaffold.registerParameter("ellipseMaxArea", 200.0,
"""The maximum area required to identify an ellipse.""")
EXPECTED_ELLIPSES_PER_FRAME = scaffold.registerParameter("expectedEllipsesPerFrame", 200,
"""The mean number of ellipses we expect to find. It's used to optimize memory
allocation.""")

class FindEllipses(scaffold.Task):
    """
    Identifies ellipses in a binary image and dumps them into an ellipses
    table in the HDF5 file.
    """

    name = "Find Ellipses"
    dependencies = [im.ParseConfig, Watershed]#, ComputeForegroundMasks, IdentifyParticles]

    def isComplete(self):
        return self.context.hasNode("ellipses")

    def run(self):
        minArea = self._param(ELLIPSE_MIN_AREA)
        maxArea = self._param(ELLIPSE_MAX_AREA)
        expectedPerFrame = self._param(EXPECTED_ELLIPSES_PER_FRAME)
        #frames = self._import(IdentifyParticles, "images")
        #frames = self._import(ComputeForegroundMasks, "masks")
        frames = self._import(Watershed, "isolated")
        
        table = self.context.createTable("ellipses", EllipseTable,
                                         expectedrows=expectedPerFrame*len(frames))
        ellipse = table.row # shortcut

        # use OpenCV to identify contours -> ellipses
        for frameNum, frame in enumerate(frames):
            contours, _ = cv2.findContours(frame.astype(np.uint8), 
                                           cv2.RETR_LIST, 
                                           cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                # Need at least 5 points for ellipses
                #if len(contour) <= 5: continue
                if len(contour) <= 1: continue

                area = cv2.contourArea(contour)
                if area < minArea or area > maxArea: continue

                vx, vy, _, _ = cv2.fitLine(contour, cv.CV_DIST_L2, 0, 0.01, 0.01)
                # don't use atan2 since we don't actually know signs anyways
                angle = math.atan(vy/vx)
                
                try:
                    moments = cv2.moments(contour)
                    x = moments.m10/moments.m00
                    y = moments.m01/moments.m00
                except:
                    boundingBox = cv2.boundingRect(contour)
                    x = boundingBox[0] + boundingBox[2]/2
                    y = boundingBox[1] + boundingBox[3]/2

                radius = math.sqrt(area)

                ellipse['frame'] = frameNum
                ellipse['area'] = area
                ellipse['position'] = [x, y]
                ellipse['angle'] = angle
                ellipse['axes'] = [radius*4, radius/4]
                ellipse.append()

                # position, axes, angle = cv2.fitEllipse(contour)

                # # TODO: are these half axes or full?
                # area = math.pi/4*np.prod(axes)
                # if area < minArea or area > maxArea: continue

                # ellipse['frame'] = frameNum
                # ellipse['area'] = area
                # ellipse['position'] = position
                # ellipse['angle'] = _TO_RADIANS*angle
                # ellipse['axes'] = axes
                # ellipse.append()

        table.flush()

    def export(self):
        return dict(ellipses=self.context.node("ellipses"))

_TO_RADIANS = math.pi/180

def groupEllipsesByFrame(ellipses):
    group = []
    current = ellipses[0]['frame']
    
    for row, frame in enumerate(ellipses.col("frame")):
        if frame != current:
            yield group
            group = []
            current = frame
        
        group.append(row)


# An HDF5 table for storing tracks
class TracksTable(tb.IsDescription):
    time     = tb.Float32Col(pos=0)
    position = tb.Float32Col(pos=1, shape=2)
    velocity = tb.Float32Col(pos=2, shape=2)
    angle    = tb.Float32Col(pos=3)
    area     = tb.Float32Col(pos=4)
    
TRACK_SEARCH_RADIUS = scaffold.registerParameter("trackSearchRadius", 0.75,
"""The maximum distance to look for the next particle in a track.""")
TRACK_MEMORY = scaffold.registerParameter("trackMemory", 0,
"""The maximum number of frame to 'remember' a particle for tracking.""")

class TrackEllipses(scaffold.Task):
    """
    Track the identified ellipses across frames, generating a table of 
    velocities.
    """
    
    class Point(tracking.PointND):
        """
        Custom point subclass so that we can store the ellipse row 
        information with it.
        """
        
        def __init__(self, frame, position, row):
            tracking.PointND.__init__(self, frame, position)
            self.row = row

    name = "Track Ellipses"
    dependencies = [im.ParseConfig, FindEllipses]

    def isComplete(self):
        return self.context.hasNode("tracks")

    def run(self):
        self._ellipses = self._import(FindEllipses, "ellipses")

        info = self._import(im.ParseConfig, "info")
        self._frameDimensions = info.imageSize
        self._dt = info.dt
        self._searchRadius = self._param(TRACK_SEARCH_RADIUS)
        self._memory = self._param(TRACK_MEMORY)
        
        self._buildLevels()
        self._track()
        self._buildTable()
        
    def _buildLevels(self):
        # Each 'level' consists of a list of all of the points in a single image
        # frame
        levels = []
        pixel = self._import(im.ParseConfig, "info").pixel
        counter = 0

        ellipses = self._ellipses # shortcut
        for rows in groupEllipsesByFrame(ellipses):
            currentLevel = []
            for row in rows:
                position = pixel*ellipses[row]["position"]
                point = TrackEllipses.Point(ellipses[row]['frame'], position, counter)
                currentLevel.append(point)
                counter += 1
            levels.append(currentLevel)
        
        self._levels = levels
        
    def _track(self):
        self._links = tracking.link_full(self._levels, 
                                         self._frameDimensions, 
                                         self._searchRadius, 
                                         tracking.Hash_table, 
                                         self._memory)
        
    def _buildTable(self):
        table = self.context.createTable("tracks", TracksTable, 
                                         expectedrows=self._ellipses.nrows) # max
        track = table.row # shortcut
        
        for link in self._links:
            ellipses = [self._ellipses[p.row] for p in link.points]
            for ellipse, nextEllipse in zip(ellipses, ellipses[1:]):
                track['time']     = ellipse['frame']*self._dt
                track['position'] = ellipse['position']
                track['angle']    = ellipse['angle']
                track['area']     = ellipse['area']
                
                dt = (nextEllipse['frame'] - ellipse['frame'])*self._dt
                dp = nextEllipse['position'] - ellipse['position']
                track['velocity'] = dp/dt
                track.append()
                
        table.flush()
                
