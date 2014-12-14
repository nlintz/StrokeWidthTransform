import connected_components as cc
import numpy as np
from matplotlib import pyplot as plt
import cv2
import math
from profiler import *
import itertools
from swt import swt

class Letter(object):
  def __init__(self, letterPixels):
    self.letterPixels = letterPixels

  def distanceToLetter(self, letter):
    centerA = self.center()
    centerB = letter.center()
    return ((centerA[0]-centerB[0])**2+(centerA[1]-centerB[1])**2)**.5

  def width(self):
    xVals = map(lambda x:x[1], self.letterPixels)
    return max(xVals) - min(xVals)

  def height(self):
    yVals = map(lambda x:x[0], self.letterPixels)
    return max(yVals) - min(yVals)

  def bounds(self):
    min_y = min([y for (y,x,w) in self.letterPixels])
    max_y = max([y for (y,x,w) in self.letterPixels])
    min_x = min([x for (y,x,w) in self.letterPixels])
    max_x = max([x for (y,x,w) in self.letterPixels])
    return ((min_y, min_x), (max_y, max_x))

  def center(self):
    ((min_y, min_x), (max_y, max_x)) = self.bounds()
    center_y = (min_y+max_y) / 2.0
    center_x = (min_x+max_x) / 2.0
    return (center_y, center_x)

  def bottomLeft(self):
    ((min_y, min_x), (max_y, max_x)) = self.bounds()
    return (min_y, min_x)

  def bottomRight(self):
    ((min_y, min_x), (max_y, max_x)) = self.bounds()
    return (min_y, max_x)

  def topLeft(self):
    ((min_y, min_x), (max_y, max_x)) = self.bounds()
    return (max_y, min_x)

  def topRight(self):
    ((min_y, min_x), (max_y, max_x)) = self.bounds()
    return (max_y, max_x)

  def strokeWidth(self):
    componentColors = map(lambda x: x[2], self.letterPixels)
    return sum(componentColors)/len(self.letterPixels)

class LetterPair(object):
  def __init__(self, letterA, letterB):
    self.letterA = letterA
    self.letterB = letterB

  def letterDistance(self):
    return self.letterA.distanceToLetter(self.letterB)

  def similarComponentStrokeWidthRatio(self, threshold=1.5):
    if max(self.letterA.strokeWidth(), self.letterB.strokeWidth())/min(self.letterA.strokeWidth(), self.letterB.strokeWidth()) < threshold:
      return True
    return False

  def similarComponentHeightRatio(self, threshold=2.0):
    if (max(self.letterA.height(), self.letterB.height()) / min(self.letterA.height(), self.letterB.height())) < threshold:
      return True
    return False

  def similarComponentDistance(self, threshold=1.5):
    if self.letterDistance() < (threshold*max(self.letterA.width(), self.letterB.width())):
      return True
    return False

  def mergeLetters(self):
    return Letter(self.letterA.letterPixels + self.letterB.letterPixels)


class LetterChain(object):
  def __init__(self):
    self.letters = []
    self.hasMerged = False
    self.direction = None

  @classmethod
  def chainFromPair(cls, pair):
    letterChain = cls()
    letterChain.letters.append(pair.letterA)
    letterChain.letters.append(pair.letterB)
    letterChain.direction = math.atan2(pair.letterA.center()[0] - pair.letterB.center()[0], 
      pair.letterA.center()[1] - pair.letterB.center()[1])
    return letterChain

  def bounds(self):
    minX = None
    maxX = None
    minY = None
    maxY = None
    for letter in self.letters:
      ((min_y, min_x), (max_y, max_x)) = letter.bounds()
      if minX == None:
        minX = min_x
        maxX = max_x
        minY = min_y
        maxY = max_y
      else:
        if min_y < minY:
          minY = min_y
        if min_x < minX:
          minX = min_x
        if max_y > maxY:
          maxY = max_y
        if max_x > maxX:
          maxX = max_x
    return ((minY, minX), (maxY, maxX))

  def chainToRegion(self):
    region = []
    for letter in self.letters:
      region += letter.letterPixels
    return Letter(region)

  def mergeWithChain(self, chain):
    lettersIndex = set(self.letters)
    for elem in chain.letters:
      if elem not in lettersIndex:
        self.letters.append(elem)

  def sharesBounds(self, chain):
    (selfMinY, selfMinX), (selfMaxY, selfMaxX) = self.bounds()
    (otherMinY, otherMinX), (otherMaxY, otherMaxX) = chain.bounds()
    if selfMinX > otherMaxX or otherMinX > selfMaxX:
      return False
    if selfMinY > otherMaxY or otherMinY > selfMaxY:
      return False
    return True


class LetterCombinator(object):
  @staticmethod
  def generateLetterPairs(letters):
    letterPairs = list(itertools.combinations(letters, 2))
    return [LetterPair(x,y) for (x,y) in letterPairs]

  @staticmethod 
  def findLines(letterChains):
    lines = []
    for chain in letterChains:
      didMerge = False
      for i, line in enumerate(lines):
        if line.sharesBounds(chain) and swt.angleDifference(line.direction, chain.direction) < math.pi/2:
          didMerge = True
          lines[i].mergeWithChain(chain)
      if not didMerge:
        lines.append(chain)
    return lines

  @staticmethod
  def findAllLines(letterChains):
    lines = LetterCombinator.findLines(letterChains)
    length = len(lines)
    while True:
      lines = LetterCombinator.findLines(lines)

      if length == len(lines):
        break
      length = len(lines)
    return lines