

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

def generateSwt(img):
  img = cv2.imread('test/images/'+img+'.jpg', 0)
  ccImg = tl.TextLocalizer.findLetterChains(img, 1)
  plt.subplot(2,1,1)
  plt.imshow(ccImg)

  ccImg = tl.TextLocalizer.findLetterChains(img, -1)
  plt.subplot(2,1,2)
  plt.imshow(ccImg)
  plt.show()

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print "Usage: python localize.py <filename>"
  else:
    generateSwt(sys.argv[1])




