import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../', 'swt'))
import letterCombinator as lc
import connected_components as cc
import swt as swt
import numpy as np
from matplotlib import pyplot as plt
import cv2
import math, random
from profiler import *
import itertools
import textLocalizer as tl

def test_localizeText():
  img = cv2.imread('test/images/rab_butler.jpg', 0)
  rows, cols = img.shape

  renderer = tl.LetterRenderer()
  localizer = tl.TextLocalizer()

  plt.subplot(2,1,1)
  new_img = np.zeros((rows, cols, 3), np.uint8)
  lines = localizer.findLines(img, 1, ['size', 'borders'])
  renderer.draw_word_lines(new_img, lines)
  plt.imshow(new_img)

  plt.subplot(2,1,2)
  new_img = np.zeros((rows, cols, 3), np.uint8)
  lines = localizer.findLines(img, -1, ['size', 'borders'])
  renderer.draw_word_lines(new_img, lines)
  plt.imshow(new_img)
  plt.show()

def test_findLetters():
  img = cv2.imread('test/images/stopsign.jpg', 0)
  rows, cols = img.shape
  new_img = np.zeros((rows, cols, 3), np.uint8)

  renderer = tl.LetterRenderer()
  localizer = tl.TextLocalizer()
  letters = localizer.findLetters(img, -1, ['size', 'borders'])

  renderer.draw_letters(new_img, letters)
  for letter in letters:
    renderer.draw_letter_rect(new_img, letter)
  plt.imshow(new_img)
  plt.show()

def test_findLetterPairs():
  img = cv2.imread('test/images/uk_dance_prototype_inspired_records.jpg', 0)
  rows, cols = img.shape
  new_img = np.zeros((rows, cols, 3), np.uint8)

  renderer = tl.LetterRenderer()
  localizer = tl.TextLocalizer()
  letters = localizer.findLetters(img, -1, ['size', 'borders'])

  letterPairs = lc.LetterCombinator.generateLetterPairs(letters)
  filteredLetterPairs = localizer.filterLetterPairs(letterPairs)

  letterChains = [lc.LetterChain.chainFromPair(pair) for pair in filteredLetterPairs]
  for chain in letterChains:
    renderer.draw_letter_rect(new_img, chain.chainToRegion())
  plt.imshow(new_img)
  plt.show()

if __name__ == "__main__":
  test_localizeText()