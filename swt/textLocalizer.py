import letterCombinator as lc
import connected_components as cc
from swt import swt
import numpy as np
from matplotlib import pyplot as plt
import cv2
import math, random
from profiler import *
import itertools

class TextLocalizer(object):
  @staticmethod
  def filterLetterPairs(letterPairs):
    letterPairs = filter(lambda x: x.similarComponentStrokeWidthRatio(), letterPairs)
    letterPairs = filter(lambda x: x.similarComponentHeightRatio(), letterPairs)
    letterPairs = filter(lambda x: x.similarComponentDistance(), letterPairs)
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
  def findLines(img, direction=-1, letterFilters=('size', 'borders', 'aspect_ratio_and_diameter')):

    letters = TextLocalizer.findLetters(img, direction, letterFilters)  

    letterPairs = lc.LetterCombinator.generateLetterPairs(letters)
    filteredLetterPairs = TextLocalizer.filterLetterPairs(letterPairs)

    letterChains = [lc.LetterChain.chainFromPair(pair) for pair in filteredLetterPairs]

    lines = lc.LetterCombinator.findAllLines(letterChains)
    validLines = filter(lambda x: len(x.letters) > 2, lines)

    return validLines

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
  def draw_word_line(img, line):
    for letter in line.letters:
      LetterRenderer.draw_letter(img, letter)
    LetterRenderer.draw_letter_rect(img, line.chainToRegion())

  @staticmethod
  def draw_word_lines(img, lines):
    for line in lines:
      LetterRenderer.draw_word_line(img, line)

  @staticmethod
  def draw_letter(img, letter):
    random_color = (255*random.random(), 255*random.random(), 255*random.random())
    for pixel in letter.letterPixels:
      (y, x, w) = pixel
      img[y, x] = random_color

  @staticmethod
  def draw_letters(img, letters):
    for letter in letters:
      LetterRenderer.draw_letter(img, letter)

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
