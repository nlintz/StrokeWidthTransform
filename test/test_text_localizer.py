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
  img = cv2.imread('test/images/ComputerScienceSmall.jpg', 0)
  rows = img.shape[0]
  cols = img.shape[1]
  # renderer = tl.LetterRenderer()
  localizer = tl.TextLocalizer()
  # letters, strokeWidthTransform, letterCandidates_arr = localizer.findLetters(img, 1, True)
  # ccImg = cc.connectedComponentsToImg(strokeWidthTransform, letterCandidates_arr, rows, cols, True)
  ccImg = localizer.findLetterChains(img, 1)
  plt.subplot(2,1,1)
  plt.imshow(ccImg)
  # for letter in letters:
    # renderer.draw_letter_rect(ccImg, letter)

  # letters, strokeWidthTransform, letterCandidates_arr = localizer.findLetters(img, -1, True)
  # ccImg = cc.connectedComponentsToImg(strokeWidthTransform, letterCandidates_arr, rows, cols, True)
  ccImg = localizer.findLetterChains(img, -1)
  plt.subplot(2,1,2)
  plt.imshow(ccImg)
  plt.show()


if __name__ == "__main__":
  test_localizeText()