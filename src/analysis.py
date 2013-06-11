"""
Tasks related to analyzing particle tracks.
"""

import numpy as np
from numpy.linalg import norm
import math
import tables as tb
import images
import scaffold
import particles

NUM_GRID_CELLS = scaffold.registerParameter("numGridCells", [20, 20]
"""The number of rows and columns in the particle grid.""")

def _griddedPath(task, path):
    return "{0}{1}_{2}".format(path, *task._param(NUM_GRID_CELLS))

class GridParticles(scaffold.Task):
    """
    Creates an NxM grid and assigns a cell number to each particle at each 
    frame, which is useful for local correlations and general optimizations.
    """

    name = "Grid Particles"
    dependencies = [particles.TrackParticles, images.ParseConfig]

    def isComplete(self):
        return self.context.hasNode(self._cellsPath)

    def export(self):
        return dict(cellMap=self.contex.node(self._cellMapPath),
                    cells=self.context.node(self._cellsPath),
                    cellCenters=self.context.node(self._cellCentersPath),
                    shape=(np.prod(self._param(NUM_GRID_CELLS)),),
                    cellSize=self._cellSize)

    def run(self):
        numGridCells = self._param(NUM_GRID_CELLS)
        imageSize = self._import(images.ParseConfig, "info").imageSize
        self._cellSize = imageSize/self._numGridCells
        tracks = self._import(particles.TrackParticles, "tracks")
        
        cellMap, cellCenters = self._buildCellMap(numGridCells)
        cells = self._assignCells(tracks, cellMap)
        
        self.context.createArray(self._cellMapPath, cellMap)
        self.context.createArray(self._cellCentersPath, cellCenters)
        self.context.createArray(self._cellsPath, cells)
        self.context.flush()
        
    def _buildCellMap(self, numGridCells):
        numRows, numCols = numGridCells
        cellMap = np.empty(numGridCells, np.uint32)
        cellCenters = []
        halfSize = self._cellSize/2.0

        for row in range(numRows):
            for col in range(numCols):
                cellMap[row, col] = row*numCols + col
                cellCenters.append([i, j]*cellSize + halfSize)

        return cellMap, np.array(cellCenters, float)
        
    def _assignCells(self, tracks, cellMap):
        cells = np.empty((tracks.nrows, 1), dtype="int")
        for row, track in enumerate(tracks):
            i, j = track['position'] // self._cellSize
            cells[row] = int(cellMap[_index(i), _index(j)])
        return cells

    @property
    def _cellsPath(self): return _griddedPath(self, "cells")
    @property
    def _cellMapPath(self): return _griddedPath(self, "cellMap")
    @property
    def _cellCentersPath(self): return _griddedPath(self, "cellCenters")


class CalculateByTime(scaffold.Task):

    dependencies = [particles.TrackParticles]
    _tablePath = "OVERRIDEME"

    def run(self):
        tracks = self._import(particles.TrackParticles, "tracks")
        assert tracks.nrows > 0

        self._table = self._setupTable()
        currentTime = tracks[0]['time']
        startRow = 0

        for row, time in enumerate(tracks.col('time')):
            if time != currentTime:
                self._onTime(time, startRow, row)
                startRow = row
                currentTime = time
        self._onFrame()

        for row, track in enumerate(tracks):
            if track['time'] != currentTime:
                self._endFrame(currentTime)
                currentTime = track['time']
                self._startFrame(currentTime)
            self._onRow(row, track)

        self._table.flush()

    def _setupTable(self):
        class TimeTable(tb.IsDescription):
            time = tb.Float32Col(pos=0)
            data = self._makeDataCol()
        return self.context.createTable(self._tablePath, TimeTable)

    def _makeDataCol(self): raise NotImplemented()
    def _startFrame(self, time): raise NotImplemented()
    def _endFrame(self, time): raise NotImplemented()
    def _onRow(self, row, track): raise NotImplemented()


class GriddedField(CalculateByTime):

    dependencies = CalculateByTime.dependencies + [GridParticles]
    _tableName = "OVERRIDEME"

    def isComplete(self):
        return self.context.hasNode(self._tablePath)

    def export(self):
        return dict(field=self.context.node(self._tablePath))

    def run(self):
        self._shape = self._import(GridParticles, "shape")
        self._cells = self._import(GridParticles, "cells")
        CalculateByTime.run(self)

    def _startFrame(self, time):
        self._resetBuckets(self._shape)

    def _endFrame(self, time):
        self._appendBuckets(self._table, time)

    def _onRow(self, row, track):
        self._addToBuckets(row, track, self._cells[row])

    def _makeDataCol(self):
        return tb.Float32Col(shape=self._shape)

    def _resetBuckets(self, shape):
        self._sums = np.zeros(shape, np.float32)
        self._counts = np.zeros(shape, np.uint16)

    def _appendBuckets(self, table, time):
        self._finalizeSums()
        table.row['time'] = time
        table.row['data'] = self._sums
        table.row.append()

    def _finalizeSums(self):
        nonzero = self._counts > 0
        self._sums[nonzero] /= self._counts[nonzero]

    def _addToBuckets(self, row, track, cell):
        self._counts[cell] += 1

    @property
    def _tablePath(self): return _griddedPath(self, self._tableName)

def _index(i): 
    return max([int(i), 0])

class NormalizeVelocities(scaffold.Task):

    name = "Normalize Velocities"
    dependencies = [particles.TrackParticles]

    def isComplete(self):
        return self.context.hasNode("normalizedVelocities")

    def export(self):
        return dict(velocities=self.context.node("normalizedVelocities"))

    def run(self):
        tracks = self._import(particles.TrackParticles, "tracks")
        normalized = np.empty((tracks.nrows, 2), float)
        for row, track in enumerate(tracks):
            normalized[row, :] = _toUnit(track['velocity'])
        self.context.createArray("normalizedVelocities", normalized)


class CalculateDensityField(GriddedField):

    name = "Calculate Density Field"
    dependencies = GriddedField.dependencies + []
    _tableName = "densityField"

    def _finalizeSums(self):
        area = np.prod(self._import(GridParticles, "cellSize"))
        self._sums = self._counts/area


class CalculateVelocityField(GriddedField):

    name = "Calculate Velocity Field"
    dependencies = GriddedField.dependencies + []
    _tableName = "velocityField"

    def _makeDataCol(self):
        return GriddedField._makeDataCol(self, self._shape + (2,))

    def _resetBuckets(self, shape):
        GriddedField._resetBuckets(self, shape + (2,))

    def _addToBuckets(self, row, track, cell):
        GriddedField._addToBuckets(self, track, cell)
        self._sums[cell, :] += track['velocity']

    def _finalizeSums(self):
        nonzero = self._counts > 0
        self._sums[nonzero, :] /= self._counts[nonzero]


class CalculateLocalVelocityCorrelationField(GriddedField):

    name = "Calculate Local Velocity Correlation Field"
    dependencies = GriddedField.dependencies + [NormalizeVelocities]
    _tableName = "localVelocityCorrelationField"

    def run(self):
        self._velocities = self._import(NormalizeVelocities, "velocities")
        GriddedField.run(self)

    def _resetBuckets(self, shape):
        self._buckets = [[] for _ in range(np.prod(shape))]

    def _addToBuckets(self, row, track, cell):
        self._buckets[cell].append(self._velocities[row])

    def _appendBuckets(self, table, time):
        table.row['time'] = time
        table.row['data'] = map(_vectorCorrelation, self._buckets)
        table.row.append()

RADII = scaffold.registerParameter("radii", np.arrange(1.0, 15, 1)
"""The successive radii to calculate correlations at.""")

class CorrelateByRadius(CalculateByTime):

    dependencies = CalculateByTime.dependencies + []

    def run(self):
        self._radii = self._param(RADII)
        self._data = np.zeros_like(self.radii, float)

    def _makeDataCol(self):
        return tb.Float32Col(shape=self._radii.shape)

    def _startFrame(self, time):
        self._data.fill(0.0)

    def _endFrame(self, time):
        table.row['time'] = time
        table.row['data'] = self._data
        table.row.append()

    def _onRow(self, row, track):
        pass # TODO grouping by frame, row range maybe?


def _toUnit(vector):
    mag = norm(vector)
    return vector/mag if mag > 1e-8 else 0*vector

def _toDirector(angle):
    return np.array([np.cos(angle), np.sin(angle)])

def _vectorCorrelation(unitVectors):
    n = len(vectors)
    if n < 2: return 0.0
    
    total = sum(np.dot(unitVectors[i], unitVectors[j]),
                for i in range(n-1)
                for j in range(i+1, n))
    # n*(n-1)/2 is the total number of dot prods
    return total/(n*(n-1)/2) 