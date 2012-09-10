

import numpy as np
import math

class Rectangle(object):
    """Rectangle that operates on numpy arrays."""

    @staticmethod
    def fromCenter(self, center, dimensions):
        half = dimensions/2
        return Rectangle(center - half, center + half)


    def __init__(self, topLeft, bottomRight):
        self.topLeft = topLeft
        self.bottomRight = bottomRight

    @property
    def dimensions(self): return self.bottomRight - self.topLeft

    @property
    def center(self): return self.topLeft + self.dimensions/2

    @property
    def left(self): return self.topLeft[0]

    @property
    def top(self): return self.topLeft[1]

    @property
    def width(self): return self.bottomRight[0] - self.topLeft[0]

    @property
    def height(self): return self.bottomRight[1] - self.topLeft[1]

    @property
    def bottom(self): return self.bottomRight[1]

    @property
    def right(self): return self.bottomRight[0]

    def intersectsWith(self, other):
        return (self.topLeft < other.bottomRight).all() and \
               (self.bottomRight > other.topLeft).all()

    @property
    def area(self): return self.dimensions.prod()

    def entirelyContains(self, other):
        return (other.topLeft > self.topLeft).all() and \
               (other.bottomRight < self.bottomRight).all()

    def merge(self, other):
        left = min(self.left, other.left)
        top = min(self.top, other.top)
        right = max(self.right, other.right)
        bottom = max(self.bottom, other.bottom)
        return Rectangle(np.array((left, top)), np.array((right, bottom)))

class QuadTree(object):

    @staticmethod
    def make(self, dimensions, goalPerRegion=2):
        region = Rectangle(np.array([0., 0.]), np.array(dimensions))
        return QuadTree(None, region, goalPerRegion)

    def __init__(self, parent, region, goalPerRegion):
        self.parent = parent
        self.region = region
        self.goalPerRegion = goalPerRegion
        
        self.particles = []
        self.children = None

    @property
    def isLeaf(self): return self.children is None

    @property
    def isRoot(self): return self.parent is None

    def add(self, particle):
        if self.isLeaf:
            self.particles.append(particle)

            if len(self.particles) > self.goalPerRegion:
                self._createChildren()

                old = self.particles
                self.particles = []
                for p in old: self._placeInChild(p)
        else:
            self._placeInChild(particle)

    def _createChildren(self):
        dim = self.region.dimensions/2
        goal = self.goalPerRegion
        topLeft = self.region.topLeft

        self._topLeft = QuadTree(self, Rectangle(topLeft, dim), goal)
        self._topRight = QuadTree(self, Rectangle(topLeft + (dim[0], 0), dim), goal)
        self._bottomLeft = QuadTree(self, Rectangle(topLeft + (0, dim[1]), dim), goal)
        self._bottomRight = QuadTree(self, Rectangle(topLeft + dim, dim), goal)
        self.children = [self._topLeft, self._topRight, 
                         self._bottomLeft, self._bottomRight]

    def _placeInChild(self, particle):
        for child in self.children:
            if child.region.entirelyContains(particle.bounds):
                child.add(particle)
                return

        self.particle.append(child)

class GridMap(object):
    """
    Fits rectangular particles into a grid of cells.
    particle.bounds must be a Rectangle!
    """

    def __init__(self, dimensions, cellSize):
        self.dimensions = np.array(dimensions)
        self.cellSize = np.array(cellSize)
        self.numCells = np.ceil(self.dimensions/self.cellSize).astype(int)
        self._cellToParticles = dict()

    def add(self, particle):
        col1, row1 = np.floor(particle.bounds.topLeft/self.cellSize)\
                     .astype(int)
        col2, row2 = np.floor(particle.bounds.bottomRight/self.cellSize)\
                     .astype(int)

        for col in range(col1, col2 + 1):
            for row in range(row1, row2 + 1):
                cell = col + row*self.numCells[0]
                self._cellToParticles.setdefault(cell, []).append(particle)

    def iterateCellGroups(self, numCells=4.0):
        sideLength = int(math.sqrt(numCells))

        for topCol in range(self.numCells[0] - sideLength):
            for leftRow in range(self.numCells[1] - sideLength):
                particles = []
                for colOffset in range(sideLength):
                    for rowOffset in range(sideLength):
                        cell = topCol + colOffset + \
                               (leftRow + rowOffset)*self.numCells[0]
                        try:
                            particles.extend(self._cellToParticles[cell])
                        except KeyError:
                            pass # no particle in cell
                yield particles




    









