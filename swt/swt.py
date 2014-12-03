""" 
Python Stroke Width Transform Implementation
Developed by Diana Vermilya and Nathan Lintz
"""

import numpy as np
import cv2
import math
import copy
from multiprocessing import Pool
from functools import partial
from profiler import *

t = Timer()
def strokeWidthTransform(img, direction=1, cannyThresholds=(100,300)):
  """ Returns the stroke width transform of a color image

  arguments --
  img: 2d grayscale array of image
  direction: -1 detects light text on dark background, 1 detects dark text on light background

  return --
  2d grayscale array where each pixel's value  is its stroke width 
  """
  edges = cv2.Canny(img, 100, 300)
  thetas = gradient(img, edges)
  firstPass, rays = castRays(edges, thetas, direction)
  
  if rays == None:
    return firstPass
  
  secondPass = refineRays(firstPass, rays)
  return secondPass

def refineRays(swt, rays):
  """ Second pass as described in article
    pixels who are longer than the median value of the ray are set to the medianLength
    arguments -- 
    swt: swt from first pass
    rays: array of pixels in ray

    returns:
    refined swt image

  """
  swt = copy.deepcopy(swt) # TODO: Do we need a deepcopy here?
  for ray in rays:
    medianLength = np.median(map(lambda x: swt[x[0]][x[1]], ray))
    for pixel in ray:
      if swt[pixel[0]][pixel[1]] > medianLength:
        swt[pixel[0]][pixel[1]] = medianLength
  return swt

def castProcess(angles, edgeLookup, maxRayLength, direction, pixel):
  ray = castRay(pixel, angles, edgeLookup, maxRayLength, direction)
  if ray:
    if len(ray) > 1:
      return ray
  return None


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

  NUM_PROC = 4
  pool = Pool(processes=NUM_PROC)
  cp = partial(castProcess, angles, edgeLookup, maxRayLength, direction)
  t.start('multiprocess')
  results = pool.map(cp, edgeIndices, len(edgeIndices)/NUM_PROC)
  print len(filter(lambda x:x!= None, results))
  t.stop('multiprocess')

  t.start('single process')
  for (row, column) in edgeIndices:
    ray = castRay((row,column), angles, edgeLookup, maxRayLength, direction)
    if ray:
      if len(ray) > 1:
        rays.append(ray)
  print len(rays)
  t.stop('single process')

  allRayLengths = map(lambda x: rayLength(x), filter(lambda x: x != None, rays))
  
  if len(allRayLengths) == 0:
    return [swt, None]

  minL, maxL = min(allRayLengths), max(allRayLengths)
  for ray in rays:
    for pixel in ray:
      swt[pixel[0], pixel[1]] = min(normalize(rayLength(ray), minL, maxL, 0, 255), swt[pixel[0], pixel[1]])
  return [swt, rays]


def normalize(value, oldMin, oldMax, newMin, newMax):
  """ interpolation function from http://stackoverflow.com/questions/929103/convert-a-number-range-to-another-range-maintaining-ratio
  arguments -- 
  value: value you are mapping from
  oldmin, oldmax: extrema of domain
  newmin, newmax: extrema of range

  return --
  value mapped to new range
  """
  return (((value - oldMin) * (newMax - newMin)) / (oldMax - oldMin)) + newMin

def castRay(startPixel, angles, edgeIndices, maxRayLength, direction):
  """ Returns length of the ray
  arguments --
  startPixel: (row, column) rays starting position
  angles: result of sobel operator
  edgeIndices: indices of edge pixels in image
  maxRayLength: maximum length of ray in pixels
  direction: 1 or -1

  return -- 
  an array of pixels if valid ray or None
  """
  row, column = startPixel
  height, width = angles.shape
  rayLength = 1
  rayDirection = angles[row][column]
  rayValid = False
  ray = [(row, column)]
  while rayLength < maxRayLength:
    pixel = (int(row + math.sin(rayDirection)*rayLength*direction), 
      int(column+math.cos(rayDirection)*rayLength*direction))
    if pixel[0] >= height or pixel[0] < 0 or pixel[1] >= width or pixel[1] < 0:
      return None

    if not rayValid:
      rayValid = True
    ray.append(pixel)

    if pixel in edgeIndices:
      oppositeDirection = angles[pixel[0]][pixel[1]]

      if angleDifference(rayDirection, oppositeDirection) > math.pi / 2:
        rayValid = False

      if rayValid:
        return ray
      else:
        return None
    
    rayLength += 1
  return None


def angleDifference(angle1, angle2):
  """ Returns angle difference between ray starting angle and the ending angle
  arguments --
  angle1: angle in radians
  angle2: angle in radians

  return -- 
  distance between the angles in radians
  """
  return abs(abs(angle1 - angle2) - math.pi)

def rayLength(ray):
  """ Returns length of the ray
  arguments --
  ray: ray of pixels

  return -- 
  ray length
  """
  return ((ray[0][0] - ray[-1][0])**2+(ray[0][1] - ray[-1][1])**2)**.5

def gradient(img, edges):
  """ Returns matrix of angles 
  arguments -- 
  edges: black and white image result of canny edge detector

  return --
  matrix of theta values
  """

  rows = np.size(img, 0)
  columns = np.size(img, 1)
  dx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize = 5, scale = -1, delta = 1, borderType = cv2.BORDER_DEFAULT)
  dy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize = 5, scale = -1, delta = 1, borderType = cv2.BORDER_DEFAULT)

  theta = np.zeros(img.shape)
  for row in range(rows):
    for col in range(columns):
      if(edges[row][col] > 0):
        theta[row][col] = math.atan2(dy[row][col], dx[row][col])
  return theta
