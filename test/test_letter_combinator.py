import sys, os
sys.path.insert(0, '../')
import lib.letterCombinator as lc
import lib.connected_components as cc
import lib.swt as swt
import numpy as np
from matplotlib import pyplot as plt
import cv2
import math, random
from lib.profiler import *
import itertools

def test_find_letters():
  img = cv2.imread('images/race_for_life.jpg', 0)
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
  img = cv2.imread('images/rab_butler.JPG', 0)
  rows, cols = img.shape

  # Compute SWT
  swt_pos = swt.strokeWidthTransform(img, -1)
  
  SAMPLE = swt_pos

  # Generate Regions
  regions = cc.connectComponents(SAMPLE)
  regions_dict = regions_to_dict(regions)
  bounds = cc.map_to_bounds(regions_dict)

  # Filter Letter Candidates
  letterCandidates_dict = cc.applyFilters(regions_dict, bounds, ['size', 'borders'])
  letterCandidates_arr = filter(lambda x: len(x) > 0, regions_to_arr(letterCandidates_dict))

  letters = [lc.Letter(x) for x in letterCandidates_arr]
  letters = filter(lambda x:x.height()>0 and x.width()>0, letters)

  # Combine Letters
  letterPairs = lc.LetterCombinator.generateLetterPairs(letters)

  letterPairs = filter(lambda x: x.similarComponentStrokeWidthRatio(2.5), letterPairs)
  letterPairs = filter(lambda x: x.similarComponentHeightRatio(), letterPairs)
  letterPairs = filter(lambda x: x.similarComponentDistance(2.0), letterPairs)

  ccImg = cc.connectedComponentsToImg(SAMPLE, letterCandidates_arr, rows, cols, True)

  letterChains = [lc.LetterChain.chainFromPair(pair) for pair in letterPairs]
  lines = lc.LetterCombinator.findAllLines(letterChains)

  for chain in lines:
    if len(chain.letters) > 2:
      draw_letter_rect(ccImg, chain.chainToRegion())

  plt.imshow(ccImg)
  plt.show()

def test_letter_pairs():
  img = cv2.imread('images/rab_butler.jpg', 0)
  rows, cols = img.shape

  # Compute SWT
  swt_pos = swt.strokeWidthTransform(img, -1)
  
  SAMPLE = swt_pos

  # Generate Regions
  regions = cc.connectComponents(SAMPLE)
  regions_dict = regions_to_dict(regions)
  bounds = cc.map_to_bounds(regions_dict)

  # Filter Letter Candidates
  letterCandidates_dict = cc.applyFilters(regions_dict, bounds, ['size', 'borders'])
  letterCandidates_arr = filter(lambda x: len(x) > 0, regions_to_arr(letterCandidates_dict))

  letters = [lc.Letter(x) for x in letterCandidates_arr]
  letters = filter(lambda x:x.height()>0 and x.width()>0, letters)

  # Combine Letters
  print len(letters)
  letterPairs = lc.LetterCombinator.generateLetterPairs(letters)

  letterPairs = filter(lambda x: x.similarComponentStrokeWidthRatio(2.5), letterPairs)
  letterPairs = filter(lambda x: x.similarComponentHeightRatio(), letterPairs)
  letterPairs = filter(lambda x: x.similarComponentDistance(2.0), letterPairs)

  # plt.subplot(2,1,1)
  letterChains = [lc.LetterChain.chainFromPair(pair) for pair in letterPairs]
  # ccImg = cc.connectedComponentsToImg(SAMPLE, letterCandidates_arr, rows, cols, True)
  # for chain in letterChains:
  #   draw_letter_rect(ccImg, chain.chainToRegion())
  # plt.imshow(ccImg)

  plt.subplot(2,1,2)
  ccImg = cc.connectedComponentsToImg(SAMPLE, letterCandidates_arr, rows, cols, True)
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

if __name__ == "__main__":
  test_letterCombinator()