
import numpy as np
cimport numpy as np
from libcpp cimport bool
cimport cython
import math

cdef extern from "math.h":
    float fabs(float x)
    float cosf(float theta)
    float sinf(float theta)
    float acosf(float theta)
    float logf(float x)

cpdef float g = 256

cpdef object mergeStatisticalRegions(np.ndarray[np.uint8_t, ndim=2] image, 
                                     float degreeSegmentation):
    cdef int height = image.shape[0] # rows
    cdef int width = image.shape[1] # columns
    cdef float delta = 1.0/(6*width*height)
    cdef float factor = g*g/2/degreeSegmentation
    cdef float logDelta = 2.0*math.log(6*width*height)

    cdef np.ndarray[np.uint8_t, ndim=1] pixels = image.flatten()
    cdef np.ndarray[float, ndim=1] average = pixels.astype(np.float32)
    cdef np.ndarray[int, ndim=1] count = np.ones(width*height, dtype=np.int32)

    cdef np.ndarray[int, ndim=1] regionIndex, nextNeighbor, neighborBucket 
    regionIndex, nextNeighbor, neighborBucket = initializeNeighbors(pixels, width, height)
    mergeAllNeighbors(regionIndex, nextNeighbor, neighborBucket, count,
                      average, factor, logDelta, width)

    consolidateAverage(average, regionIndex)
    cdef int numRegions = consolidateRegions(regionIndex)

    return (numRegions, 
            np.reshape(average, (height, width)), 
            np.reshape(regionIndex, (height, width)))

@cython.boundscheck(False)
cdef consolidateAverage(np.ndarray[float, ndim=1] average, 
                        np.ndarray[int, ndim=1] regionIndex):
    cdef int i
    for i in range(len(average)):
        average[i] = average[getRegionIndex(i, regionIndex)]

@cython.boundscheck(False)
cdef int consolidateRegions(np.ndarray[int, ndim=1] regionIndex):
    cdef int numRegions = 0
    cdef int i
    for i in range(len(regionIndex)):
        if regionIndex[i] < 0:
            regionIndex[i] = regionIndex[-1 - regionIndex[i]]
        else:
            regionIndex[i] = numRegions
            numRegions += 1
    return numRegions

@cython.boundscheck(False)
cdef mergeAllNeighbors(np.ndarray[int, ndim=1] regionIndex, 
                       np.ndarray[int, ndim=1] nextNeighbor, 
                       np.ndarray[int, ndim=1] neighborBucket,
                       np.ndarray[int, ndim=1] count,
                       np.ndarray[float, ndim=1] average,
                       float factor,
                       float logDelta,
                       int width):
    cdef int i, neighborIndex, r1, r2
    cdef float difference, f1, f2

    for i in range(len(neighborBucket)):
        neighborIndex = neighborBucket[i]
        while neighborIndex >= 0:
            r1 = neighborIndex//2
            r2 = r1 + (1 if neighborIndex & 1 == 0 else width)
            
            r1 = getRegionIndex(r1, regionIndex)
            r2 = getRegionIndex(r2, regionIndex)

            difference = average[r1] - average[r2]
            f1 = getLogFactor(count[r1], logDelta)
            f2 = getLogFactor(count[r2], logDelta)

            if difference**2 < factor*(f1 + f2):
                mergeRegions(r1, r2, count, average, regionIndex)

            neighborIndex = nextNeighbor[neighborIndex]

cdef float getLogFactor(float count, float logDelta):
    return (logf(1 + count)*minFloat(g, count) + logDelta)/count

@cython.boundscheck(False)
cdef int getRegionIndex(int index, np.ndarray[int, ndim=1] regionIndex):
    index = regionIndex[index]
    while index < 0:
        index = regionIndex[-1 - index]
    return index

@cython.boundscheck(False)
cdef mergeRegions(int r1, int r2, 
                  np.ndarray[int, ndim=1] count, 
                  np.ndarray[float, ndim=1] average, 
                  np.ndarray[int, ndim=1] regionIndex):
    if r1 == r2: return

    cdef int mergedCount = count[r1] + count[r2]
    cdef float mergedAverage = (average[r1]*count[r1] + 
                                average[r2]*count[r2])/mergedCount

    # always merge to smaller index
    if r1 > r2:
        r1, r2 = r2, r1

    average[r1] = mergedAverage
    count[r1] = mergedCount
    regionIndex[r2] = -1 - r1

@cython.boundscheck(False)
cdef object initializeNeighbors(np.ndarray[np.uint8_t, ndim=1] pixels,
                                int width, int height):
    cdef np.ndarray[int, ndim=1] regionIndex, nextNeighbor, neighborBucket
    regionIndex = np.arange(width*height, dtype=np.int32)
    nextNeighbor = np.zeros(2*width*height, dtype=np.int32)
    neighborBucket = -1*np.ones(256, dtype=np.int32)

    # initialize neighbors
    cdef int i, j, index, neighborIndex
    for j from height > j >= 0:
        for i from width > i >= 0:
            index = i + width*j
            neighborIndex = 2*index

            # vertical
            if j < height - 1:
                addNeighborPair(nextNeighbor, neighborBucket,
                                neighborIndex + 1, pixels[index], pixels[index + width])

            # horizontal
            if i < width - 1:
                addNeighborPair(nextNeighbor, neighborBucket,
                                neighborIndex, pixels[index], pixels[index + 1])

    return regionIndex, nextNeighbor, neighborBucket

@cython.boundscheck(False)
cdef addNeighborPair(np.ndarray[int, ndim=1] nextNeighbor, 
                     np.ndarray[int, ndim=1] neighborBucket, 
                     int neighborIndex, int p1, int p2):
    cdef int difference = abs(p1 - p2)
    nextNeighbor[neighborIndex] = neighborBucket[difference]
    neighborBucket[difference] = neighborIndex

cdef float minFloat(float x, float y):
    return x if x < y else y








