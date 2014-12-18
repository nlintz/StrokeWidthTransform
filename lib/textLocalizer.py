import letterCombinator as lc
import connected_components as cc
import swt
import numpy as np
from matplotlib import pyplot as plt
import cv2
import math, random
from profiler import *
import itertools

class TextLocalizer(object):
  @staticmethod
  def filterLetterPairs(letterPairs, strokeThreshold=1.5, heightThreshold=2, distanceThreshold=1.5):
    letterPairs = filter(lambda x: x.similarComponentStrokeWidthRatio(strokeThreshold), letterPairs)
    letterPairs = filter(lambda x: x.similarComponentHeightRatio(heightThreshold), letterPairs)
    letterPairs = filter(lambda x: x.similarComponentDistance(distanceThreshold), letterPairs)
    return letterPairs

  @staticmethod
  def findLetters(img, direction, letterFilters):
    strokeWidthTranform = swt.strokeWidthTransform(img, direction)
    # Generate Regions
    regions = cc.connectComponents(strokeWidthTranform)
    regions_dict = TextLocalizer.regions_to_dict(regions)
    bounds = cc.map_to_bounds(regions_dict)
    # Filter Letter Candidates
    letterCandidates_dict = cc.applyFilters(regions_dict, bounds, letterFilters)
    letterCandidates_arr = filter(lambda x: len(x) > 0, TextLocalizer.regions_to_arr(letterCandidates_dict))
    letters = [lc.Letter(x) for x in letterCandidates_arr]
    letters = filter(lambda x: x.height() > 0 and x.width() > 0, letters)

    return letters

  @staticmethod
  def findLines(img, direction=-1, letterFilters=('size', 'borders'), **kwargs):

    letters = TextLocalizer.findLetters(img, direction, letterFilters)  

    letterPairs = lc.LetterCombinator.generateLetterPairs(letters)
    filteredLetterPairs = TextLocalizer.filterLetterPairs(letterPairs, **kwargs)

    letterChains = [lc.LetterChain.chainFromPair(pair) for pair in filteredLetterPairs]

    lines = lc.LetterCombinator.findAllLines(letterChains)
    # validLines = filter(lambda x: len(x.letters) > 2, lines)
    validLines = TextLocalizer.validateLines(lines)
    return validLines

  @staticmethod
  def validateLines(lines, heightThreshold=2.0):
    linesWithEnoughLetters = filter(lambda x: len(x.letters) > 2, lines)
    validLines = []

    return linesWithEnoughLetters

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
  def draw_word_line(img, line, letterColor=None, rectColor=None):
    # for letter in line.letters:
      # LetterRenderer.draw_letter(img, letter, letterColor)
    LetterRenderer.draw_letter_rect(img, line.chainToRegion(), rectColor)

  @staticmethod
  def draw_word_lines(img, lines, letterColor=None, rectColor=None):
    for line in lines:
      LetterRenderer.draw_word_line(img, line, letterColor, rectColor)

  @staticmethod
  def draw_letter(img, letter, color=None):
    if color==None:
      color = (255*random.random(), 255*random.random(), 255*random.random())
    for pixel in letter.letterPixels:
      (y, x, w) = pixel
      img[y, x] = color

  @staticmethod
  def draw_letters(img, letters, color=None):
    for letter in letters:
      LetterRenderer.draw_letter(img, letter)

  @staticmethod
  def draw_letter_rect(img, letter, color=None):
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

    if color == None:
      color = (255*random.random(), 255*random.random(), 255*random.random())
    strokeWidth = 4
    cv2.line(img, lower_left, lower_right, color, strokeWidth)
    cv2.line(img, lower_left, upper_left, color, strokeWidth)
    cv2.line(img, upper_right, lower_right, color, strokeWidth)
    cv2.line(img, upper_right, upper_left, color, strokeWidth)

  @staticmethod
  def draw_letter_center(img, letter):
    cv2.circle(img, (int(letter.center()[1]), int(letter.center()[0])), 2, (255, 0, 0))
