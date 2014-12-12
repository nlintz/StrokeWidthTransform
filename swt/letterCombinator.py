import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../', 'swt'))
import swt.swt as swt
import swt.connected_components as cc
import numpy as np
from matplotlib import pyplot as plt
import cv2
import math
from profiler import *
import itertools
import swt.swt as swt

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

  def head(self):
    return self.letters[0]

  def tail(self):
    return self.letters[-1]

  def chainToRegion(self):
    region = []
    for letter in self.letters:
      region += letter.letterPixels
    return Letter(region)

  def make_merge_comparator(self,head):
    def compare(x, y):
        if x.distanceToLetter(head) > y.distanceToLetter(head):
            return 1
        elif x.distanceToLetter(head) < y.distanceToLetter(head):
            return -1
        else:
            return 0
    return compare

  def mergeWithChain(self, chain):
    for elem in chain.letters:
      if elem not in self.letters:
        self.letters.append(elem)

    self.letters = sorted(self.letters, cmp=self.make_merge_comparator(self.head()), reverse=True)
    self.hasMerged = True
    chain.hasMerged = True

  def sharesEnd(self, chain): # TODO - Change to SharesBoundingBox
    if self.head() == chain.head():
      return True
    if self.head() == chain.tail():
      return True
    if self.tail() == chain.head():
      return True
    if self.tail() == chain.tail():
      return True
    return False

  def sharesBounds(self, chain):
    for letter in chain.letters:
      if letter in self.letters:
        return True
    return False

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
      lettersIndex = set(chain.letters)
      for i in range(len(lines)):
        for letter in lines[i].letters:
          if letter in lettersIndex:
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


  # @staticmethod
  # def findConnectedChains(letterChains):
  #   if len(letterChains) == 1:
  #     return (letterChains, False)

  #   didMerge = False
  #   connectedChains = []

  #   for i in range(len(letterChains)):
  #     for j in range(len(letterChains)):
  #       if i != j:
  #         chainA = letterChains[i]
  #         chainB = letterChains[j]
  #         if not chainA.hasMerged and not chainB.hasMerged:
  #           if chainA.sharesBounds(chainB) and swt.angleDifference(chainA.direction, chainB.direction) < math.pi/2.0:
  #             didMerge = True
  #             chainA.mergeWithChain(chainB)
  #             connectedChains.append(chainA)
  #           if len(chainA.letters) > 2:
  #             connectedChains.append(chainA)
  #           if len(chainB.letters) > 2:
  #             connectedChains.append(chainB)

  #   connectedChains = filter(lambda x:len(x.letters) > 2, connectedChains)

  #   for chain in connectedChains:
  #     chain.hasMerged = False
  #   return (connectedChains, didMerge)

  # @staticmethod
  # def findLines(letterChains):
  #   foundAllChains = False
  #   while not foundAllChains:
  #     letterChains, didMerge = LetterCombinator.findConnectedChains(letterChains)
  #     if not didMerge:
  #       foundAllChains = True
  #   return letterChains



