import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../', 'swt'))
import letterCombinator as lc
import swt.connected_components as cc
import swt.swt as swt
import numpy as np
from matplotlib import pyplot as plt
import cv2
import math, random
from profiler import *
import itertools

class TextLocalizer(object):
  @staticmethod
  def findLetters(img, direction, returnswt=False):
    rows, cols = img.shape
    # Compute SWT
    strokeWidthTranform = swt.strokeWidthTransform(img, direction)
    # Generate Regions
    regions = cc.connectComponents(strokeWidthTranform)
    regions_dict = TextLocalizer.regions_to_dict(regions)
    bounds = cc.map_to_bounds(regions_dict)

    # Filter Letter Candidates
    letterCandidates_dict = cc.applyFilters(regions_dict, bounds, ['size', 'borders', 'aspect_ratio_and_diameter'])
    letterCandidates_arr = filter(lambda x: len(x) > 0, TextLocalizer.regions_to_arr(letterCandidates_dict))
    letters = [lc.Letter(x) for x in letterCandidates_arr]

    if returnswt:
      return (letters, strokeWidthTranform, letterCandidates_arr)
    return letters

  @staticmethod
  def findLetterChains(img, direction=-1):
    rows, cols = img.shape

    letters, strokeWidthTranform, letterCandidates_arr = TextLocalizer.findLetters(img, direction, True)  
    print len(letters)
    ccImg = cc.connectedComponentsToImg(strokeWidthTranform, letterCandidates_arr, rows, cols, True)

    letterPairs = lc.LetterCombinator.generateLetterPairs(letters)
    letterPairs = filter(lambda x: x.similarComponentStrokeWidthRatio(), letterPairs)
    letterPairs = filter(lambda x: x.similarComponentHeightRatio(), letterPairs)
    letterPairs = filter(lambda x: x.similarComponentDistance(), letterPairs)

    letterChains = [lc.LetterChain.chainFromPair(pair) for pair in letterPairs]

    lines = lc.LetterCombinator.findAllLines(letterChains)
    for chain in lines:
      if len(chain.letters) > 2:
        LetterRenderer.draw_letter_rect(ccImg, chain.chainToRegion())
    return ccImg

  @staticmethod
  def regions_to_dict(regions):
    d = {}
    for i, region in enumerate(regions):
      d[i] = region
    return d

  @staticmethod
  def regions_to_arr(regions):
    arr = [[] for i in range(max(regions.keys())+1)]
    for i, v in regions.items():
      arr[i] = v
    return arr


class LetterRenderer(object):

  @staticmethod
  def draw_letter_rect(img, letter):
    # if len(region) == 0:
    #   return
    rows, cols, _ = img.shape
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

  @staticmethod
  def draw_letter_center(img, letter):
    cv2.circle(img, (int(letter.center()[1]), int(letter.center()[0])), 2, (255, 0, 0))
