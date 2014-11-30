""" 
Python Stroke Width Transform Implementation
Developed by Diana Vermilya and Nathan Lintz
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
from scipy.interpolate import interp1d

def strokeWidthTransform(img, cannyThresholds=(100, 200)):
  """ Returns the stroke width transform of a color image

  arguments --
  img: 2d grayscale array of image

  return --
  2d grayscale array where each pixel's value  is its stroke width 
  """
  swt = np.empty((img.shape[0], img.shape[1]))
  swt.fill(255) # swt vector is initialized with maximum values
  # Apply Canny Edge Detection to input
  edges = cv2.Canny(img, *cannyThresholds)
  rays = findStrokeWidthRays(edges)
  pixels = getStrokeWidthPixels(rays)

  # pixels = filterPixels(pixels, rays)
  normalized = interp1d([min(pixels.values()), max(pixels.values())], [0, 255])
  for (row,column), width in pixels.items():
    if width:
      swt[row, column] = normalized(width)

  return swt

def angleDifference(angle1, angle2):
  """ Returns shortest distance between two angles
  arguments --
  angle1: angle in radians
  angle2: angle in radians

  return -- 
  shortest distance between the angles in radians e.g. (2*pi - .01, 0) -> .01
  """
  return math.atan2(math.sin(angle1-angle2), math.cos(angle1-angle2))


def castRay(edges, angles, startPixel, antigradient=False):
  """ Returns an array of pixels if the ray is valid e.g.
    on the image and whose endpoint has an opposite direction 
    to the start point

  arguments --
  edges: 2d grayscale array from canny edge detector
  angles: 2d array of pixel direction angle
  startPixel: (row,column) tuple for the starting point of the ray
  antigradient: Boolean over whether you want to cast the ray in the 
    same direction as the gradient or the opposite direction

  return --
  array of (row, column, strokeWidth) pixels
  """
  ray = [startPixel]
  row, column = startPixel
  strokeWidth = 1
  angle = angles[row, column]
  
  if antigradient:
    angle += math.pi;
  while True:
    column = startPixel[1] + int(round(strokeWidth * math.cos(angle)))
    row = startPixel[0] - int(round(strokeWidth * math.sin(angle)))
    if row >= edges.shape[0] or row < 0 or column >= edges.shape[1] or column < 0:
      return None
    
    ray.append((row, column))
    if edges[row, column] == edges[startPixel[0], startPixel[1]]:
      break
    strokeWidth += 1

  if len(ray):
    strokeWidth = rayLength(ray)
  ray = [pixel + (strokeWidth,) for pixel in ray]
  if rayValid(ray, angles):
    return ray

  else:
    return None

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

def gradient(img):
  """ Returns an np 3d matrix mapping pixels to gradient angle
  arguments --
  img: 2d grayscale array of image

  return --
  2d array of pixel direction angle

  """
  sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize = 5, scale = -1)
  sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize = 5, scale = -1)

  return cv2.phase(sobelx, sobely)
  # return np.arctan2(np.uint8(np.absolute(sobely)), np.uint8(np.absolute(sob
  # return np.arctan2(sobely, sobelx)
  # theta = np.zeros(img.shape, np.float32)
      
  # # calculating theta
  # height = np.size(img, 0)
  # width = np.size(img, 1)
  # for row in range(height):
  #   for col in range(width):
  #     if(img[row][col] > 0):
  #       theta[row][col] = math.atan2(sobely[row][col], sobelx[row][col])
  # return theta


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