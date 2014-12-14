import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../', 'swt'))
import letterCombinator as lc
import connected_components as cc
from swt import swt
import numpy as np
from matplotlib import pyplot as plt
import cv2
import math, random
from profiler import *
import itertools

def test_find_letters():
  img = cv2.imread('test/images/emergency_stop.jpg', 0)
  rows, cols = img.shape

  # Compute SWT
  swt_pos = swt.strokeWidthTransform(img, 1)
  
  SAMPLE = swt_pos

  # Generate Regions
  regions = cc.connectComponents(SAMPLE)
  regions_dict = regions_to_dict(regions)
  bounds = cc.map_to_bounds(regions_dict)

  # Filter Letter Candidates
  # letterCandidates_dict = cc.applyFilters(regions_dict, bounds, ['size', 'borders', 'aspect_ratio_and_diameter'])
  letterCandidates_dict = cc.applyFilters(regions_dict, bounds, ['size', 'borders'])
  letterCandidates_arr = filter(lambda x: len(x) > 0, regions_to_arr(letterCandidates_dict))

  letters = [lc.Letter(x) for x in letterCandidates_arr]

  for letter in letters:
    draw_letter_rect(swt_pos, letter)
  plt.imshow(swt_pos, 'gray')
  plt.show()

def test_letterCombinator():
  img = cv2.imread('test/race_for_life.jpg', 0)
  rows, cols = img.shape

  # Compute SWT
  swt_pos = swt.strokeWidthTransform(img, 1)
  swt_pos_dilated = 255 - cv2.dilate(255 - swt_pos, kernel = np.ones((2,2),np.uint8), iterations = 1)
  
  SAMPLE = swt_pos

  # Generate Regions
  regions = cc.connectComponents(SAMPLE)
  regions_dict = regions_to_dict(regions)
  bounds = cc.map_to_bounds(regions_dict)

  # Filter Letter Candidates
  letterCandidates_dict = cc.applyFilters(regions_dict, bounds, ['size', 'borders', 'aspect_ratio_and_diameter'])
  letterCandidates_arr = filter(lambda x: len(x) > 0, regions_to_arr(letterCandidates_dict))

  letters = [lc.Letter(x) for x in letterCandidates_arr]

  # Combine Letters
  letterPairs = lc.LetterCombinator.generateLetterPairs(letters)

  letterPairs = filter(lambda x: x.similarComponentStrokeWidthRatio(), letterPairs)
  letterPairs = filter(lambda x: x.similarComponentHeightRatio(), letterPairs)
  letterPairs = filter(lambda x: x.similarComponentDistance(), letterPairs)

  ccImg = cc.connectedComponentsToImg(SAMPLE, letterCandidates_arr, rows, cols, True)

  letterChains = [lc.LetterChain.chainFromPair(pair) for pair in letterPairs]

  lines = lc.LetterCombinator.findAllLines(letterChains)
  for chain in lines:
    if len(chain.letters) > 2:
      draw_letter_rect(ccImg, chain.chainToRegion())

  plt.imshow(ccImg)
  plt.show()

def regions_to_dict(regions):
  d = {}
  for i, region in enumerate(regions):
    d[i] = region
  return d

def regions_to_arr(regions):
  arr = [[] for i in range(max(regions.keys())+1)]
  for i, v in regions.items():
    arr[i] = v
  return arr

def draw_letter_rect(img, letter):
  # if len(region) == 0:
  #   return
  if len(img.shape) == 3:
    rows, cols, _ = img.shape
  else:
    rows, cols = img.shape
  box = letter.bounds()
  (miny, minx), (maxy, maxx) = box
  lower_left = (minx, miny)
  lower_right = (maxx, miny)
  upper_left = (minx, maxy)
  upper_right = (maxx, maxy)

  color = (255*random.random(), 255*random.random(), 255*random.random())

  cv2.line(img, lower_left, lower_right, color, 2)
  cv2.line(img, lower_left, upper_left, color, 2)
  cv2.line(img, upper_right, lower_right, color, 2)
  cv2.line(img, upper_right, upper_left, color, 2)

def draw_letter_center(img, letter):
  cv2.circle(img, (int(letter.center()[1]), int(letter.center()[0])), 2, (255, 0, 0))


if __name__ == "__main__":
  test_find_letters()