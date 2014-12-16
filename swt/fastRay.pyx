import math
import numpy as np

cdef float angleDifference(float angle1, float angle2):
  """ Returns angle difference between ray starting angle and the ending angle
  arguments --
  angle1: angle in radians
  angle2: angle in radians

  return -- 
  distance between the angles in radians
  """
  return abs(abs(angle1 - angle2) - math.pi)

def castRays(edges, angles, direction, maxRayLength=100):
  """ casts a ray for every edge in the image
  arguments --
  edges: black and white image result of canny edge detector
  angles: black and white image result of sobel operator
  direction: 1 or -1

  return -- 
  [swt first pass, rays]
  """
  swt = np.empty((edges.shape[0], edges.shape[1]))
  swt.fill(255) # swt vector is initialized with maximum values
  rays = []
  nonZeroEdges = edges.nonzero()
  edgeIndices = zip(nonZeroEdges[0], nonZeroEdges[1])
  edgeLookup = set(edgeIndices)

  for (row, column) in edgeIndices:
    ray = castRay(row, column, angles, edgeLookup, maxRayLength, direction)
    if ray:
      if len(ray) > 1:
        rays.append(ray)

  allRayLengths = map(lambda x: rayLength(x), filter(lambda x: x != None, rays))
  
  if len(allRayLengths) == 0:
    return [swt, None]

  minL, maxL = min(allRayLengths), max(allRayLengths)
  for ray in rays:
    for pixel in ray:
      swt[pixel[0], pixel[1]] = min(normalize(rayLength(ray), minL, maxL, 0, 255), swt[pixel[0], pixel[1]])
  return [swt, rays]

cdef castRay(int startRow, int startColumn, angles, edgeIndices, int maxRayLength, int direction):
  """ Returns length of the ray
  arguments --
  row/column: ray starting position
  angles: result of sobel operator
  edgeIndices: indices of edge pixels in image
  maxRayLength: maximum length of ray in pixels
  direction: 1 or -1

  return -- 
  an array of pixels if valid ray or None
  """

  cdef int height = angles.shape[0]
  cdef int width = angles.shape[1]
  cdef int rayLength = 1
  cdef float rayDirection = angles[row][column]
  cdef int rayValid = False
  cdef int rayPixelRow
  cdef int rayPixelColumn
  cdef int oppositeDirection
  cdef int row = startRow
  cdef int column = startColumn

  ray = [(startRow, startColumn)]

  while rayLength < maxRayLength:

    rayPixelRow = <int>row + math.sin(rayDirection)*rayLength*direction
    rayPixelColumn = <int>column+math.cos(rayDirection)*rayLength*direction
    if rayPixelRow >= height or rayPixelRow < 0 or rayPixelColumn >= width or rayPixelColumn < 0:
      return None

    if not rayValid:
      rayValid = True
    ray.append((rayPixelRow, rayPixelColumn))

    if (rayPixelRow, rayPixelColumn) in edgeIndices:
      oppositeDirection = angles[rayPixelRow][rayPixelColumn]

      if angleDifference(rayDirection, oppositeDirection) > (math.pi / 2):
        rayValid = False

      if rayValid:
        return ray
      else:
        return None
    
    rayLength += 1
  return None

cdef float normalize(float value, int oldMin, int oldMax, int newMin, int newMax):
  """ interpolation function from http://stackoverflow.com/questions/929103/convert-a-number-range-to-another-range-maintaining-ratio
  arguments -- 
  value: value you are mapping from
  oldmin, oldmax: extrema of domain
  newmin, newmax: extrema of range

  return --
  value mapped to new range
  """
  # return value
  return (((value - oldMin) * (newMax - newMin)) / (oldMax - oldMin)) + newMin

cdef float rayLength(ray):
  """ Returns length of the ray
  arguments --
  ray: ray of pixels

  return -- 
  ray length
  """
  return ((ray[0][0] - ray[-1][0])**2+(ray[0][1] - ray[-1][1])**2)**.5
