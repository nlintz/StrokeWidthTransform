""" 
Python Stroke Width Transform Implementation
Developed by Diana Vermilya and Nathan Lintz
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

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
  strokes = findStrokes(edges)
  return swt

def castRay(edges, angles, startPixel, antigradient=False):
  """ Returns an array of pixels if the ray is valid e.g.
    on the image and whose endpoint has an opposite direction 
    to the start point

  arguments --
  edges: 2d grayscale array from canny edge detector
  angles: 2d array of pixel direction angle
  startPixel: (x,y) tuple for the starting point of the ray
  antigradient: Boolean over whether you want to cast the ray in the 
    same direction as the gradient or the opposite direction

  return --
  array of (y, x, strokeWidth) pixels
  """
  ray = [startPixel]
  y, x = startPixel
  strokeWidth = 1
  angle = angles[y, x]
  
  if antigradient:
    angle += math.pi;

  while True:
    x = x + int(strokeWidth * math.cos(angle))
    y = y - int(strokeWidth * math.sin(angle))
    
    if y >= img.shape[0] or y < 0 or x >= img.shape[1] or x < 0:
      return None
    
    if edges[y, x] != 0:
      break
    
    strokeWidth += 1

  ray = [pixel + (strokeWidth,) for pixel in ray]
  if rayValid(ray, angles):
    return ray

  else:
    return None

def findStrokes(edges):
  """ Returns an array of valid Rays where each ray is 
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
  for i,row in enumerate(edges):
    for j,column in enumerate(row):
      if edges[i, j] > 0:
        rayWithGradient = castRay(edges, angles, (i,j))
        if rayWithGradient != None:
          validRays.append(rayWithGradient)
      
        rayAgainstGradient = castRay(edges, angles, (i,j), True)
        if rayAgainstGradient != None:
          validRays.append(rayAgainstGradient)
  
  return validRays

def gradient(img):
  """ Returns an np 3d matrix mapping pixels to gradient angle
  arguments --
  img: 2d grayscale array of image

  return --
  2d array of pixel direction angle

  """
  sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
  sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
  return cv2.phase(sobelx, sobely)

def rayValid(ray, angles):
  """ Returns boolean for whether or not the cast ray is valid
  arguments --
  ray: array of pixels
  angles: 2d array of pixel direction angle

  return --
  True if array is valid False otherwise

  """
  maxWidth = angles.shape[0]
  maxHeight = angles.shape[1]

  startpoint = ray[0]
  endpoint = ray[-1]
  
  if abs(angles[startpoint[0], startpoint[1]]) - abs(angles[endpoint[0], endpoint[1]]) > math.pi / 6.0:
    return False

  for pixel in ray:
    x = pixel[0]
    y = pixel[1]
    width = pixel[2]
    if x >= maxWidth or x < 0:
      return False
    if y >= maxHeight or y < 0:
      return False
  return True

img = cv2.imread('stopsign.jpg',0)

plt.imshow(strokeWidthTransform(img), cmap="gray")
plt.show()
