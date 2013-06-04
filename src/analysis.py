"""
Tasks related to analyzing particle tracks.
"""

import numpy as np
import math
import tables as tb
import images
import scaffold
import particles

_numGridCells = scaffold.registerParameter("numGridCells", [20, 20]
"""The number of rows and columns in the particle grid.""")

def getCellNumbersPath(numGridCells):
    return "cellNumbers{0}_{1}".format(*numGridCells)
    
def _index(i): 
    return max([int(i), 0])

class GridParticles(scaffold.Task):
    """
    Creates an NxM grid and assigns a cell number to each particle at each 
    frame, which is useful for local correlations and general optimizations.
    """

    name = "Grid Particles"
    dependencies = [particles.TrackParticles, images.ParseConfig]

    def isComplete(self):
        return data.hasNode(getCellNumbersPath(data.params[_numGridCells]))

    def copy(self):
        return GridParticles()

    def _run(self, data):
        self._numGridCells = data.params[_numGridCells]
        self._cellSize = data.attrs.imageSize/numGridCells
        self._tracks = data.root[processing.TRACKS_PATH]
        
        self._buildCellMap()
        self._assignCells()
        
        data.createArray(getCellNumbersPath(self._numGridCells), self._cells)
        data.flush()
        
    def _buildCellMap(self):
        numRows, numCols = self._numGridCells
        cellMap = np.empty(self._numGridCells, dtype="int")
        for row in range(numRows):
            for col in range(numCols):
                cellMap[row, col] = row*numCols + col
        self._cellMap = cellMap
        
    def _assignCells(self):
        cells = np.empty((self._tracks.nrows, 1), dtype="int")
        for row, track in enumerate(self._tracks):
            i, j = track['position'] // self._cellSize
            cells[row] = int(self._cellMap[_index(i), _index(j)])
        self._cells = cells




