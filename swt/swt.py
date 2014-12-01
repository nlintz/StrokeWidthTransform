""" 
Python Stroke Width Transform Implementation
Developed by Diana Vermilya and Nathan Lintz
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import random
from scipy.interpolate import interp1d

def strokeWidthTransform(img, direction=-1, cannyThresholds=(100,300)):
  """ Returns the stroke width transform of a color image

  arguments --
  img: 2d grayscale array of image
  direction: -1 detects light text on dark background, 1 detects dark text on light background

  return --
  2d grayscale array where each pixel's value  is its stroke width 
  """  
  edges = cv2.Canny(img, 100, 300)
  thetas = gradient(img, edges)
  plt.subplot(2,1,1)
  plt.imshow(edges, 'gray')

  firstPass, rays = castRays(edges, thetas)

  # Debug Plotting
  plt.subplot(2,1,2)
  plt.imshow(firstPass, 'gray')
  plt.show()



def castRays(edges, angles, maxRayLength=200):
  swt = np.empty((edges.shape[0], edges.shape[1]))
  swt.fill(255) # swt vector is initialized with maximum values
  rays = []
  nonZeroEdges = edges.nonzero()
  edgeIndices = zip(nonZeroEdges[0], nonZeroEdges[1])
  edgeLookup = set(edgeIndices)
  for (row, column) in edgeIndices:
    rays.append(castRay((row,column), angles, edgeLookup, maxRayLength))

  allRayLengths = map(lambda x: rayLength(x), filter(lambda x: x != None, rays))
  normalized = interp1d([min(allRayLengths), max(allRayLengths)], [0, 255])
  for ray in rays:
    if ray:
      if len(ray) > 1:
        for pixel in ray:
          swt[pixel[0], pixel[1]] = min(normalized(rayLength(ray)), swt[pixel[0], pixel[1]])
  return [swt, rays]


def castRay((row, column), angles, edgeIndices, maxRayLength):
  height, width = angles.shape
  rayLength = 1
  rayDirection = angles[row][column]
  rayValid = False
  ray = [(row, column)]
  while rayLength < maxRayLength:
    pixel = (int(row + math.sin(rayDirection)*rayLength), int(column+math.cos(rayDirection)*rayLength))
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
