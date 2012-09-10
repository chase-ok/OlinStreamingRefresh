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
import utils
from scipy import stats
from trackpy import tracking

from matplotlib import pyplot as plt

ELLIPSE_MIN_AREA = scaffold.registerParameter("ellipseMinArea", 4.0,
"""The minimum area required to identify an ellipse.""")
ELLIPSE_MAX_AREA = scaffold.registerParameter("ellipseMaxArea", 200.0,
"""The maximum area required to identify an ellipse.""")
EXPECTED_ELLIPSES_PER_FRAME = scaffold.registerParameter("expectedEllipsesPerFrame", 200,
"""The mean number of ellipses we expect to find. It's used to optimize memory
allocation.""")

ELLIPSE_TABLE_PATH = "ellipses"

class ParticleFinder(scaffold.Task):
    """Base class for all of the different ways to find particles."""

    def _findEllipsesViaContours(self, masks):
        minArea = self._param(ELLIPSE_MIN_AREA)
        maxArea = self._param(ELLIPSE_MAX_AREA)
        expectedPerFrame = self._param(EXPECTED_ELLIPSES_PER_FRAME)
        
        table = self.context.createTable(ELLIPSE_TABLE_PATH, EllipseTable,
                                         expectedrows=expectedPerFrame*len(masks))
        ellipse = table.row # shortcut

        # use OpenCV to identify contours -> ellipses
        for frameNum, mask in enumerate(masks):
            contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                           cv2.RETR_LIST, 
                                           cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                # Need at least 5 points for ellipses
                #if len(contour) <= 5: continue
                if len(contour) <= 1: continue

                area = cv2.contourArea(contour)
                if area < minArea or area > maxArea: continue

                ellipse['frame'] = frameNum
                ellipse['position'] = estimatePositionFromContour(contour)
                ellipse['angle'] = estimateAngleFromContour(contour)
                ellipse['axes'] = estimateEllipseAxesFromContour(contour)
                ellipse.append()

        table.flush()

class FindParticlesViaRegions(ParticleFinder):
    """
    Identifies ellipses in a binary image and dumps them into an ellipses
    table in the HDF5 file.
    """

    name = "Find Ellipses"
    dependencies = [im.ParseConfig, im.MergeStatisticalRegions]

    def isComplete(self):
        return self.context.hasNode(ELLIPSE_TABLE_PATH)

    def run(self):
        masks = self._import(im.MergeStatisticalRegions, "masks")
        self._findEllipsesViaContours(masks)

    def export(self):
        return dict(ellipses=self.context.node(ELLIPSE_TABLE_PATH))


DILATION_RADIUS = scaffold.registerParameter("dilationRadius", 3,
"""Amount (in pixels) to dilate each image.""")
MORPH_THRESHOLD = scaffold.registerParameter("morphThreshold", 0.2,
"""Threshold for pixel values (0-1) after morphological procedures.""")
BLUR_SIGMA = scaffold.registerParameter("blurSigma", 2.0,
"""The std. dev. (in pixels) for the gaussian blur.""")
EXP_THRESHOLD = scaffold.registerParameter("expThreshold", 0.0001,
"""Threshold for pixel values after final dilation.""")

class FindParticlesViaMorph(ParticleFinder):
    """
    Identify particles using a combination of foreground masks and the 
    algorithm described at https://github.com/tacaswell/tracking
    """

    name = "Find Particles via Morphological Operations"
    dependencies = [im.LoadImages, im.ComputeForegroundMasks]

    def run(self):
        self._images = self._import(im.LoadImages, "images")
        self._masks = self._import(im.ComputeForegroundMasks, "masks")
        self._imageSize = self._images[0].shape

        self._featureRadius = self._param(im.FEATURE_RADIUS)
        self._dilationRadius = self._param(DILATION_RADIUS)
        self._morphThreshold = self._param(MORPH_THRESHOLD)
        self._blurSigma = self._param(BLUR_SIGMA)
        self._expThreshold = self._param(EXP_THRESHOLD)

        # apply the foreground masks to the original images
        #self._images = [i*m.astype(np.uint8) for i, m in zip(self._images, self._masks)]

        self._doMorph()
        self._findLocalMax()
        self._findEllipsesViaContours(self._maxed)

    def export(self):
        return dict(images=im.ImageSeq(self._maxed),
                    ellipses=self.context.node(ELLIPSE_TABLE_PATH))

    def _doMorph(self):
        """Apply a gaussian blur and then a tophat."""
        tophatKernel = im.makeCircularKernel(self._featureRadius)

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
        dilationKernel = im.makeCircularKernel(self._dilationRadius)

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

# An HDF5 table for storing identified ellipses
class EllipseTable(tb.IsDescription):
    frame    = tb.UInt32Col(pos=0)
    position = tb.Float32Col(pos=1, shape=2)
    angle    = tb.Float32Col(pos=2)
    axes     = tb.Float32Col(pos=3, shape=2)

class Particle(object):

    @staticmethod
    def fromContour(contour, majorMinorAxesRatio=4.0):
        return Particle(estimatePositionFromContour(contour),
                        estimateAngleFromContour(contour),
                        estimateEllipseAxesFromContour(contour, 
                                                       majorMinorAxesRatio),
                        estimateBoundsFromContour(contour))

    def __init__(self, position, angle, axes, bounds):
        self.position = position
        self.angle = angle
        self.axes = axes
        self.area = math.pi*axes.prod()
        self.bounds = bounds

    def drawOn(self, canvas, color=(0, 0, 255), degreeDelta=20):
        points = cv2.ellipse2Poly(tuple(map(int, self.position)),
                                  tuple(map(int, self.axes)),
                                  self.angle*180/math.pi,
                                  0, 360,
                                  degreeDelta)
        cv2.fillConvexPoly(canvas, points, color, cv2.CV_AA)

PARTICLE_AXES = scaffold.registerParameter("particleAxes", [6.0, 1.5],
"""The default assumed axes for elliptical particles.""")

EDGE_THRESHOLD = scaffold.registerParameter("edgeThreshold", 200,
"""The minimum pixel value for the Sobel edge computation in particle
seeding.""")

FRAGMENT_ANGLE_MAX = scaffold.registerParameter("fragmentAngleMax", math.pi/6,
"""The maximum angle difference between two fragments for them to be considered
to be from the same particle.""")

FRAGMENT_DISTANCE_MAX = scaffold.registerParameter("fragmentDistanceMax", 2,
"""The maximum distance between two fragments for them to be considered
to be from the same particle.""")

FRAGMENT_AREA_MAX = scaffold.registerParameter("fragmentAreaMax", 0.25,
"""The maximum relative area difference between two fragments for them to be
considered to be from the same particle.""")

class FindParticlesViaEdges(scaffold.Task):

    name = "Fit Model to Images"
    dependencies = [im.ParseConfig, im.RemoveBackground]

    def run(self):
        self._images = self._import(im.RemoveBackground, "images")

        #debug
        self._seedImages = []

        perFrame = self._param(EXPECTED_ELLIPSES_PER_FRAME)
        self._table = self.context.createTable("ellipses_", EllipseTable,
                                               expectedrows=perFrame*len(self._images))
        for i, image in enumerate(self._images[:50]):
            particles = self._seedParticles(image)
            particles = self._refineParticles(particles, image)
            self._render(i, particles)
        self._table.flush()

    def _seedParticles(self, image):
        #compute edges
        def derivativeSquared(dx, dy):
            deriv = np.empty(image.shape, dtype=np.float32)
            cv2.Sobel(image, cv2.CV_32F, dx, dy, deriv, 3)
            return deriv**2
        edges = np.sqrt(derivativeSquared(1, 0) + derivativeSquared(0, 1))
        seedImage = (edges > self._param(EDGE_THRESHOLD)).astype(np.uint8)

        #despeckle
        despeckleKernel = np.array([[1, 1, 1],
                                    [1, 0, 1],
                                    [1, 1, 1]], dtype=np.uint8)
        speckles = cv2.filter2D(seedImage, -1, despeckleKernel) == 0
        seedImage[speckles] = 0

        #debug
        self._seedImages.append(seedImage*255)

        contours, _ = cv2.findContours(seedImage, 
                                       cv2.RETR_LIST, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        return map(Particle.fromContour, contours)

    def _refineParticles(self, particles, image):
        tooSmall = 0.5
        probablyFragment = 5
        probablyConjoined = 25

        # filter into categories
        refined = [] # no work necessary
        fragments = []
        conjoined = []
        for particle in particles:
            if particle.area <= tooSmall:
                pass
            elif particle.area <= probablyFragment:
                fragments.append(particle)
            elif particle.area >= probablyConjoined:
                conjoined.append(particle)
            else:
                refined.append(particle)

        refined.extend(self._mergeFragments(fragments))
        refined.extend(self._splitConjoined(conjoined))
        return refined
    
    def _mergeFragments(self, fragments):
        removed = set()
        merged = []

        # find nearby fragments using grid
        dimensions = self._import(im.ParseConfig, "info").shape
        cellSize = 2*max(self._param(PARTICLE_AXES))
        grid = utils.GridMap(dimensions, np.array([cellSize, cellSize]))
        for particle in fragments: grid.add(particle)

        angleMax = self._param(FRAGMENT_ANGLE_MAX)
        distanceMax = self._param(FRAGMENT_DISTANCE_MAX)
        areaMax = self._param(FRAGMENT_AREA_MAX)

        # go through groups to see if fragments are nearby and aligned
        for group in grid.iterateCellGroups():
            for i, p1 in enumerate(group[:-1]):
                if p1 in removed: continue

                for p2 in group[i+1:]:
                    if p2 in removed: continue

                    angleDiff = abs(p1.angle - p2.angle)
                    # adjust for angle pi -> -pi wraparounds
                    realDiff = min(2*math.pi - angleDiff, angleDiff)
                    if realDiff > angleMax: continue

                    areaDiff = abs(p1.area - p2.area)/max(p1.area, p2.area)
                    if areaDiff > areaMax: continue

                    distance = np.linalg.norm(p1.position - p2.position)
                    if distance > distanceMax: continue

                    # actually merge aligned fragments
                    major = max(p1.axes[0], p2.axes[0])
                    minor = p1.axes[1] + p2.axes[1]
                    merged.append(Particle((p1.position + p2.position)/2,
                                           p1.angle,
                                           np.array([major, minor]),
                                           p1.bounds.merge(p2.bounds)))
                    removed.add(p1)
                    removed.add(p2)

        for fragment in fragments:
            if fragment not in removed:
                merged.append(fragment)
        return merged

    def _splitConjoined(self, conjoined):
        return conjoined

    def _render(self, frameNum, particles):
        ellipse = self._table.row
        for particle in particles:
            ellipse['frame'] = frameNum
            ellipse['position'] = particle.position
            ellipse['angle'] = particle.angle
            ellipse['axes'] = particle.axes
            ellipse.append()

    def export(self):
        return dict(ellipses=self.context.node("ellipses_"),
                    seedImages=self._seedImages)

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

def estimateAngleFromContour(contour):
    vx, vy, _, _ = cv2.fitLine(contour, cv.CV_DIST_L2, 0, 0.01, 0.01)
    # don't use atan2 since we don't actually know signs anyways
    return math.atan(vy/vx)

def estimatePositionFromContour(contour):
    try:
        moments = cv2.moments(contour)
        x = moments['m10']/moments['m00']
        y = moments['m01']/moments['m00']
    except:
        bounds = cv2.boundingRect(contour)
        x = bounds[0] + bounds[2]/2
        y = bounds[1] + bounds[3]/2
    return np.array([x, y])

def estimateEllipseAxesFromContour(contour, majorMinorAxesRatio=4.0):
    area = cv2.contourArea(contour)
    major = math.sqrt(area/math.pi*majorMinorAxesRatio)
    minor = major/majorMinorAxesRatio
    return np.array([major, minor])

def estimateBoundsFromContour(contour):
    points = cv2.boundingRect(contour)
    return utils.Rectangle(np.array(points[0:2]), np.array(points[2:4]))

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

class TrackParticles(scaffold.Task):
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
    dependencies = [im.ParseConfig, FindParticlesViaEdges]

    def isComplete(self):
        return self.context.hasNode("tracks")

    def run(self):
        self._ellipses = self._import(FindParticlesViaEdges, "ellipses")

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
                point = TrackParticles.Point(ellipses[row]['frame'], position, counter)
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
                
