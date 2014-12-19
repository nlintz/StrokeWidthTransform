import sys, os
sys.path.insert(0, '../')
import lib.swt as swt
import lib.letterCombinator as lc
import lib.textLocalizer as tl
import lib.helpers as helpers
import numpy as np
from matplotlib import pyplot as plt
import cv2
import math, random
from lib.profiler import *
import itertools

def test_localizeText():
  imgName = 'images/rab_butler.JPG'
  img = cv2.imread(imgName, 0)
  imgColor = cv2.imread(imgName, 1)
  rows, cols = img.shape

  renderer = tl.LetterRenderer()
  localizer = tl.TextLocalizer()

  plt.subplot(2,1,1)
  new_img = np.zeros((rows, cols, 3), np.uint8)
  lines = localizer.findLines(img, 1, ['size', 'borders', 'aspect_ratio_and_diameter'])
  renderer.draw_word_lines(imgColor, lines)
  plt.imshow(helpers.bgr2rgb(imgColor))

  plt.subplot(2,1,2)
  imgColor = cv2.imread(imgName, 1)

  new_img = np.zeros((rows, cols, 3), np.uint8)
  lines = localizer.findLines(img, -1, ['size', 'borders', 'aspect_ratio_and_diameter'])
  renderer.draw_word_lines(imgColor, lines)
  plt.imshow(helpers.bgr2rgb(imgColor))
  plt.show()

def test_localizeText_traffic():
  imgName = 'images/traffic_large.jpg'
  img = cv2.imread(imgName, 0)
  imgColor = cv2.imread(imgName, 1)
  rows, cols = img.shape

  renderer = tl.LetterRenderer()
  localizer = tl.TextLocalizer()

  plt.subplot(2,1,1)
  new_img = np.zeros((rows, cols, 3), np.uint8)
  lines = localizer.findLines(img, 1, ['size', 'borders', 'aspect_ratio_and_diameter'], strokeThreshold=1.5, heightThreshold=2.0, distanceThreshold=1.5)
  renderer.draw_word_lines(imgColor, lines, rectColor=(0, 0, 255))
  cv2.imwrite('results/traffic/words_pos.jpg', imgColor)
  plt.imshow(helpers.bgr2rgb(imgColor))

  plt.subplot(2,1,2)
  imgColor = cv2.imread(imgName, 1)

  new_img = np.zeros((rows, cols, 3), np.uint8)
  lines = localizer.findLines(img, -1, ['size', 'borders', 'aspect_ratio_and_diameter'], strokeThreshold=1.5, heightThreshold=2.0, distanceThreshold=1.5)
  renderer.draw_word_lines(imgColor, lines, rectColor=(0, 0, 255))
  cv2.imwrite('results/traffic/words_neg.jpg', imgColor)
  plt.imshow(helpers.bgr2rgb(imgColor))
  plt.show()

def test_localizeText_rabButler():
  imgName = 'uk_dance_prototype_inspired_records.JPG'
  img = cv2.imread('images/' + imgName, 0)
  imgColor = cv2.imread('images/' + imgName, 1)
  rows, cols = img.shape

  renderer = tl.LetterRenderer()
  localizer = tl.TextLocalizer()

  plt.subplot(2,1,1)
  new_img = np.zeros((rows, cols, 3), np.uint8)
  lines = localizer.findLines(img, 1, ['size', 'borders', 'aspect_ratio_and_diameter'])
  renderer.draw_word_lines(imgColor, lines, rectColor=(255, 0, 0))
  cv2.imwrite('results/expo_pictures/results_pos_' + imgName, imgColor)
  plt.imshow(helpers.bgr2rgb(imgColor))

  plt.subplot(2,1,2)
  imgColor = cv2.imread('images/' + imgName, 1)

  lines = localizer.findLines(img, -1, ['size', 'borders', 'aspect_ratio_and_diameter'])
  renderer.draw_word_lines(imgColor, lines, rectColor=(255, 0, 0))
  cv2.imwrite('results/expo_pictures/results_neg_' + imgName, imgColor)
  plt.imshow(helpers.bgr2rgb(imgColor))
  plt.show()

def test_findLetters():
  imgName='rab_butler_large.JPG'
  img = cv2.imread('images/'+imgName, 0)
  rows, cols = img.shape

  renderer = tl.LetterRenderer()
  localizer = tl.TextLocalizer()
  letters = localizer.findLetters(img, 1, ['size', 'borders', 'aspect_ratio_and_diameter'])
  new_img = np.zeros((rows, cols, 3), np.uint8)
  renderer.draw_letters(new_img, letters)
  for letter in letters:
    renderer.draw_letter_rect(new_img, letter)
  plt.subplot(2,1,1)
  plt.imshow(new_img)
  # cv2.imwrite('results/rab_butler/letters_pos.jpg', new_img)

  letters = localizer.findLetters(img, -1, ['size', 'borders', 'aspect_ratio_and_diameter'])
  new_img = np.zeros((rows, cols, 3), np.uint8)
  renderer.draw_letters(new_img, letters)
  plt.subplot(2,1,2)
  for letter in letters:
    renderer.draw_letter_rect(new_img, letter)
  plt.imshow(new_img)
  cv2.imwrite('results/expo/localize_' + imgName, new_img)
  plt.show()

def test_findLetterPairs():
  imgName = 'images/rab_butler_large.JPG'
  img = cv2.imread(imgName, 0)
  rows, cols = img.shape
  new_img = cv2.imread(imgName, 1)

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
  test_localizeText_rabButler()