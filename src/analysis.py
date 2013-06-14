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

NUM_GRID_CELLS = scaffold.registerParameter("numGridCells", [20, 20],
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
        self._loadCellSize()
        return dict(cellMap=self.context.node(self._cellMapPath),
                    cells=self.context.node(self._cellsPath),
                    cellCenters=self.context.node(self._cellCentersPath),
                    shape=(np.prod(self._param(NUM_GRID_CELLS)),),
                    cellSize=self._cellSize)

    def run(self):
        numGridCells = self._param(NUM_GRID_CELLS)
        self._loadCellSize()
        tracks = self._import(particles.TrackParticles, "tracks")
        
        cellMap, cellCenters = self._buildCellMap(numGridCells)
        cells = self._assignCells(tracks, cellMap)
        
        self.context.createChunkArray(self._cellMapPath, cellMap)
        self.context.createChunkArray(self._cellCentersPath, cellCenters)
        self.context.createChunkArray(self._cellsPath, cells)
        self.context.flush()

    def _loadCellSize(self):
        self._imageSize = self._import(images.ParseConfig, "info").imageSize
        self._cellSize = self._imageSize/self._param(NUM_GRID_CELLS)
        
    def _buildCellMap(self, numGridCells):
        numRows, numCols = numGridCells
        cellMap = np.empty(numGridCells, np.uint32)
        cellCenters = []
        halfSize = self._cellSize/2.0

        for row in range(numRows):
            for col in range(numCols):
                cellMap[row, col] = row*numCols + col
                cellCenters.append([row, col]*self._cellSize + halfSize)

        return cellMap, np.array(cellCenters, float)
        
    def _assignCells(self, tracks, cellMap):
        cells = np.empty((tracks.nrows, 1), np.uint32)
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
    _tablePath = None

    def isComplete(self):
        print self.name + str(self.context.hasNode(self._tablePath))
        return self.context.hasNode(self._tablePath)

    def export(self):
        return dict(table=self.context.node(self._tablePath))

    def run(self):
        if self._tablePath is None: raise NotImplemented

        self._tracks = self._import(particles.TrackParticles, "tracks")
        assert self._tracks.nrows > 0

        self._table = self._setupTable()
        currentTime = self._tracks[0]['time']
        startRow = 0

        for row, time in enumerate(self._tracks.col('time')):
            if time != currentTime:
                self._onTime(time, startRow, row)
                startRow = row
                currentTime = time
        self._onTime(currentTime, startRow, self._tracks.nrows)

        self._table.flush()

    def _setupTable(self):
        class TimeTable(tb.IsDescription):
            time = tb.Float32Col(pos=0)
            data = self._makeDataCol()
        return self.context.createTable(self._tablePath, TimeTable)

    def _makeDataCol(self): raise NotImplemented
    def _onTime(self, time, startRow, stopRow): raise NotImplemented


class GriddedField(CalculateByTime):

    dependencies = CalculateByTime.dependencies + [GridParticles]

    def export(self):
        return dict(field=self.context.node(self._tablePath))

    def run(self):
        self._shape = self._import(GridParticles, "shape")
        self._cells = self._import(GridParticles, "cells")
        CalculateByTime.run(self)

    def _onTime(self, time, start, stop):
        self._resetBuckets()
        for offset, track in enumerate(self._tracks.iterrows(start, stop)):
            row = start + offset
            self._addToBuckets(row, track, self._cells[row])
        self._appendBuckets(time)

    def _makeDataCol(self):
        return tb.Float32Col(shape=self._shape)

    def _resetBuckets(self):
        self._sums = np.zeros(self._shape, np.float32)
        self._counts = np.zeros(self._shape, np.uint16)

    def _appendBuckets(self, time):
        self._finalizeSums()
        self._table.row['time'] = time
        self._table.row['data'] = self._sums
        self._table.row.append()

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
        velocities = tracks.col('velocity')
        magnitudes = np.sqrt(np.sum(velocities**2, axis=1))
        nonzero = magnitudes > 1e-8
        normalized = np.zeros_like(velocities)
        normalized[nonzero, :] = velocities[nonzero, :]\
                                 /magnitudes[nonzero][:, np.newaxis]
        self.context.createChunkArray("normalizedVelocities", normalized)


class CalculateDirectors(scaffold.Task):

    name = "Calculate Directors"
    dependencies = [particles.TrackParticles]

    def isComplete(self):
        return self.context.hasNode("directors")

    def export(self):
        return dict(directors=self.context.node("directors"))

    def run(self):
        tracks = self._import(particles.TrackParticles, "tracks")
        angles = tracks.col('angle')
        directors = np.vstack([np.cos(angles), np.sin(angles)]).T
        self.context.createChunkArray("directors", directors)


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
        return tb.Float32Col(shape=self._shape + (2,))

    def _resetBuckets(self):
        self._sums = np.zeros(self._shape + (2,), np.float32)
        self._counts = np.zeros(self._shape, np.uint16)

    def _addToBuckets(self, row, track, cell):
        GriddedField._addToBuckets(self, row, track, cell)
        self._sums[cell, :] += track['velocity']

    def _finalizeSums(self):
        nonzero = self._counts > 0
        self._sums[nonzero, :] /= self._counts[nonzero][:, np.newaxis]


class CalculateLocalVelocityCorrelationField(GriddedField):

    name = "Calculate Local Velocity Correlation Field"
    dependencies = GriddedField.dependencies + [NormalizeVelocities]
    _tableName = "localVelocityCorrelationField"

    def run(self):
        self._velocities = self._import(NormalizeVelocities, "velocities")
        GriddedField.run(self)

    def _resetBuckets(self):
        self._buckets = [[] for _ in range(np.prod(self._shape))]

    def _addToBuckets(self, row, track, cell):
        self._buckets[cell].append(self._velocities[row])

    def _appendBuckets(self, time):
        self._table.row['time'] = time
        self._table.row['data'] = map(self._vectorCorrelation, self._buckets)
        self._table.row.append()

    def _vectorCorrelation(self, vectors):
        n = len(vectors)
        if n < 2: return 0.0
        
        total = sum(np.dot(vectors[i], vectors[j])
                    for i in range(n-1)
                    for j in range(i+1, n))
        # n*(n-1)/2 is the total number of dot prods
        return total/(n*(n-1)/2) 


RADII = scaffold.registerParameter("radii", np.arange(1.0, 15, 1),
"""The successive radii to calculate correlations at.""")

class ComputeCircleAreas(scaffold.Task):

    name = "Compute Circle Areas"
    dependencies = [images.ParseConfig]

    def isComplete(self):
        if not self.context.hasNode("_circleAreasRadii"): return False

        radii = self._param(RADII)
        match = self.context.node("_circleAreasRadii")
        # TODO: Allow subsets instead of strict matches
        return len(radii) == len(match) and (np.abs(radii-match) < 0.1).all()

    def export(self):
        self._table = self.context.node("_circleAreas")
        self._loadShape()
        self._areas = self._table.cols.areas
        return dict(table=self._table,
                    circleArea=self._getArea)

    def run(self):
        self._radii = self._param(RADII)
        self.context.createArray("_circleAreasRadii", self._radii)
        self._loadShape()

        self._makeTable()
        for radius in self._radii:
            self._table.row['radius'] = radius
            self._table.row['areas'] = self._computeForRadius(radius)
            self._table.row.append()
        self._table.flush()

    def _loadShape(self):
        self._imageSize = self._import(images.ParseConfig, "info").imageSize
        self._shape = (self._imageSize/2 + 1).astype(int)

    def _makeTable(self):
        class AreasTable(tb.IsDescription):
            radius = tb.Float32Col(pos=0)
            areas = tb.Float32Col(shape=self._shape)
        self._table = self.context.createTable("_circleAreas", AreasTable, 
                                               expectedrows=len(self._radii))
        self._table.cols.radius.createCSIndex()

    def _computeForRadius(self, r):
        areas = np.empty(self._shape, np.float32)

        for x in range(self._shape[1]):
            clipLeft = x - r < 0
            for y in range(self._shape[0]):    
                clipTop = y - r < 0

                if clipLeft and clipTop:
                    area = self._clipCorner(x, y, r)
                elif clipLeft and not clipTop:
                    area = self._clipSide(x, r)
                elif not clipLeft and clipTop:
                    area = self._clipSide(y, r)
                else:
                    area = math.pi*r**2
                areas[y, x] = area

        return areas

    def _clipCorner(self, x, y, r):
        x2, y2, r2 = x**2, y**2, r**2
            
        if x2 + y2 < r2:
            xIntercept = x + math.sqrt(r2 - y2)
            yIntercept = y + math.sqrt(r2 - x2)
            angle = 2*math.asin(math.sqrt(xIntercept*yIntercept)/(2*r))
            triangle = 0.5*xIntercept*yIntercept
            chord = 0.5*r2*(angle - math.sin(angle))
            return triangle + chord
        else:
            return x*math.sqrt(r2 - x2) + y*math.sqrt(r2 - y2) + \
                   r2*(math.asin(x/r) + math.asin(y/r))

    def _clipSide(self, x, r):
        return x*math.sqrt(r**2 - x**2) + r**2*(math.pi/2 + math.asin(x/r))

    def _getArea(self, x, y, radiusIndex):
        x, y = int(x), int(y)
        if x >= self._shape[1]: x = 2*self._shape[1] - x - 1
        if y >= self._shape[0]: y = 2*self._shape[0] - y - 1
        return self._areas[radiusIndex][y, x]


class CorrelateByRadius(CalculateByTime):

    dependencies = CalculateByTime.dependencies + [ComputeCircleAreas]

    def run(self):
        self._radii = self._param(RADII)
        self._data = np.zeros_like(self._radii, float)
        self._circleArea = self._import(ComputeCircleAreas, "circleArea")
        CalculateByTime.run(self)

    def _makeDataCol(self):
        return tb.Float32Col(shape=self._radii.shape)

    def _onTime(self, time, start, stop):
        self._data.fill(0)

        tracks = self._tracks.read(start, stop)
        positions = tracks["position"]
        info = dict(time=time, start=start, stop=stop, tracks=tracks)

        for offset, track in enumerate(tracks):
            info['offset'] = offset
            info['track'] = track

            position = track['position']
            distancesSquared = ((positions-position)**2).sum(axis=1)

            for i, rOuter in enumerate(self._radii):
                rInner = 1e-8 if i == 0 else self._radii[i-1]

                info['inAnnulus'] = (distancesSquared >= rInner**2) &\
                                    (distancesSquared <= rOuter**2)
                if len(info['inAnnulus']) == 0: continue

                result = self._correlate(info)
                if result:
                    x, y = position
                    area = self._circleArea(x, y, i)
                    if i != 0: area -= self._circleArea(x, y, i-1)
                    self._data[i] += result/area

        self._data /= len(tracks) # computing the average

        self._table.row['time'] = time
        self._table.row['data'] = self._data
        self._table.row.append()

    def _correlate(self, info): raise NotImplemented


class CorrelateParticleDistance(CorrelateByRadius):

    name = "Correlate Particle Distance"
    _tablePath = "particleDistanceCorrelation"

    def _correlate(self, info):
        return info['inAnnulus'].sum()


class CorrelateVelocities(CorrelateByRadius):

    name = "Correlate Velocities"
    dependencies = CorrelateByRadius.dependencies + [NormalizeVelocities]
    _tablePath = "velocityCorrelation"

    def run(self):
        self._velocities = self._import(NormalizeVelocities, "velocities")
        CorrelateByRadius.run(self)

    def _onTime(self, time, start, stop):
        self._currentVelocities = self._velocities[start:stop, :]
        CorrelateByRadius._onTime(self, time, start, stop)

    def _correlate(self, info):
        velocities = self._currentVelocities[info['inAnnulus'], :]
        total = np.dot(velocities, 
                       self._currentVelocities[info['offset']]).sum()
        return total/len(info['inAnnulus'])


class CorrelateDirectors(CorrelateByRadius):

    name = "Correlate Director"
    dependencies = CorrelateByRadius.dependencies + [CalculateDirectors]
    _tablePath = "directorCorrelation"

    def run(self):
        self._directors = self._import(CalculateDirectors, "directors")
        CorrelateByRadius.run(self)

    def _onTime(self, time, start, stop):
        self._currentDirectors = self._directors[start:stop, :]
        CorrelateByRadius._onTime(self, time, start, stop)

    def _correlate(self, info):
        directors = self._currentDirectors[info['inAnnulus'], :]
        total = np.dot(directors, self._currentDirectors[info['offset']]).sum()
        return total/len(info['inAnnulus'])


class CorrelateDirectorVelocities(CorrelateByRadius):

    name = "Correlate Director-Velocities"
    dependencies = CorrelateByRadius.dependencies + \
                   [CalculateDirectors, NormalizeVelocities]
    _tablePath = "directorVelocitiesCorrelation"

    def run(self):
        self._directors = self._import(CalculateDirectors, "directors")
        self._velocities = self._import(NormalizeVelocities, "velocities")
        CorrelateByRadius.run(self)

    def _onTime(self, time, start, stop):
        self._currentVelocities = self._velocities[start:stop, :]
        CorrelateByRadius._onTime(self, time, start, stop)

    def _correlate(self, info):
        total = np.dot(self._currentVelocities[info['inAnnulus'], :],
                       self._directors[info['start'] + info['offset']]).sum()
        return total/len(info['inAnnulus'])
