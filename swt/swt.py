""" 
Python Stroke Width Transform Implementation
Developed by Diana Vermilya and Nathan Lintz
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
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

  firstPass = castRays(edges, thetas)
  for ray in firstPass:
    if ray:
      for pixel in ray:
        edges[pixel[0], pixel[1]] = 100
  
  plt.imshow(edges, 'gray')
  plt.show()

  # rays = findStrokeWidthRays(theta)
  # pixels = getStrokeWidthPixels(rays)

  # # pixels = filterPixels(pixels, rays)
  # normalized = interp1d([min(pixels.values()), max(pixels.values())], [0, 255])
  # for (row,column), width in pixels.items():
  #   if width:
  #     swt[row, column] = normalized(width)

  # return swt

def castRays(edges, angles, maxRayLength=100):
  swt = np.empty((edges.shape[0], edges.shape[1]))
  swt.fill(255) # swt vector is initialized with maximum values
  rays = []
  nonZeroEdges = edges.nonzero()
  edgeIndices = zip(nonZeroEdges[0], nonZeroEdges[1])
  edgeLookup = set(edgeIndices)
  for (row, column) in edgeIndices:
    rays.append(castRay((row,column), angles, edgeLookup, maxRayLength))
  return rays

def castRay((row, column), angles, edgeIndices, maxRayLength):
  height, width = angles.shape
  rayLength = 1
  rayDirection = angles[row][column]
  rayValid = False
  ray = []
  while rayLength < maxRayLength:
    pixel = (int(row + math.sin(rayDirection)*rayLength), int(column+math.cos(rayDirection)*rayLength))
    if pixel[0] >= height or pixel[0] < 0 or pixel[1] >= width or pixel[1] < 0:
      return None

    if not rayValid:
      rayValid = True
    
    if pixel in edgeIndices:
      if rayValid:
        return ray
      else:
        return None

    ray.append(pixel)
    
    rayLength += 1
  return None


def angleDifference(angle1, angle2):
  """ Returns shortest distance between two angles
  arguments --
  angle1: angle in radians
  angle2: angle in radians

  return -- 
  shortest distance between the angles in radians e.g. (2*pi - .01, 0) -> .01
  """
  return math.atan2(math.sin(angle1-angle2), math.cos(angle1-angle2))


# def castRay(edges, angles, startPixel, antigradient=False):
#   """ Returns an array of pixels if the ray is valid e.g.
#     on the image and whose endpoint has an opposite direction 
#     to the start point

#   arguments --
#   edges: 2d grayscale array from canny edge detector
#   angles: 2d array of pixel direction angle
#   startPixel: (row,column) tuple for the starting point of the ray
#   antigradient: Boolean over whether you want to cast the ray in the 
#     same direction as the gradient or the opposite direction

#   return --
#   array of (row, column, strokeWidth) pixels
#   """
#   ray = [startPixel]
#   row, column = startPixel
#   strokeWidth = 1
#   angle = angles[row, column]
  
#   if antigradient:
#     angle += math.pi;
#   while True:
#     column = startPixel[1] + int(round(strokeWidth * math.cos(angle)))
#     row = startPixel[0] - int(round(strokeWidth * math.sin(angle)))
#     if row >= edges.shape[0] or row < 0 or column >= edges.shape[1] or column < 0:
#       return None
    
#     ray.append((row, column))
#     if edges[row, column] == edges[startPixel[0], startPixel[1]]:
#       break
#     strokeWidth += 1

#   if len(ray):
#     strokeWidth = rayLength(ray)
#   ray = [pixel + (strokeWidth,) for pixel in ray]
#   if rayValid(ray, angles):
#     return ray

#   else:
#     return None

def filterPixels(pixels, rays):
  """ Returns a new set of pixels filtered for weird letter cases e.g. bottoms of L

  """
  filteredPixels = {}
  for ray in rays:
    row, column, minWidth = min(ray, key=lambda x:x[2])
    for row, column, width in ray:
      filteredPixels[(row,column)] = minWidth
  return filteredPixels


def findStrokeWidthRays(edges):
  """ Returns an array of valid rays where each ray is 
    an array of pixels whose grayscale value is their
    stroke width

  arguments --
  edges: 2d grayscale array from canny edge detector

  return --
  array of rays where a ray is an array of pixels

  """
  # Compute edge gradients
  angles = gradient(edges)
  validRays = []

  # Cast a ray in the direction of the edge gradient for each pixel
  for rowIndex, row in enumerate(edges):
    for columnIndex, column in enumerate(row):
      if edges[rowIndex, columnIndex] == 255:
        rayWithGradient = castRay(edges, angles, (rowIndex,columnIndex))
        rayAgainstGradient = castRay(edges, angles, (rowIndex,columnIndex), True)

        if rayWithGradient and rayAgainstGradient:
          validRays.append(min(rayWithGradient, rayAgainstGradient, key=lambda x: rayLength(x)))

        elif rayWithGradient != None:
          validRays.append(rayWithGradient)
        elif rayAgainstGradient != None:
          validRays.append(rayAgainstGradient)
  return validRays

def getStrokeWidthPixels(rays):
  """ Returns a dictionary mapping pixels to their stroke widths.
    The width of each pixel is the minimum width of any pixel in its row,column location

    arguments --
    rays: array of (row, column, strokeWidth) pixels

    return --
    dictionary mapping (row,column) -> stroke width
  """
  strokeWidthPixels = {}
  for ray in rays:
    for row, column, width in ray:
      if (row,column) in strokeWidthPixels:
        strokeWidthPixels[(row,column)] = min(strokeWidthPixels[(row,column)], width)
      else:
        strokeWidthPixels[(row,column)] = width
  return strokeWidthPixels

def rayLength(ray):
  return ((ray[0][0] - ray[-1][0])**2+(ray[0][1] - ray[-1][1])**2)**.5

def rayValid(ray, angles):
  """ Returns boolean for whether or not the cast ray is valid
  arguments --
  ray: array of pixels
  angles: 2d array of pixel direction angle

  return --
  True if array is valid False otherwise

  """
  if len(ray) < 2:
    return False

  startpoint = ray[0]
  endpoint = ray[-1]
  if angleDifference(angles[startpoint[0], startpoint[1]], angles[endpoint[0], endpoint[1]]) > math.pi / 6.0:
    return False
  return True

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
