import letterCombinator as lc
import textLocalizer as tl
import numpy as np
from matplotlib import pyplot as plt
import cv2
import math, random
from profiler import *
import itertools

class TextCropper(object):
  @staticmethod
  def cropTextRegionsFromImage(img):
    """ returns an array of cropped images which likely contain text regions
    """
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = imgGray.shape
    localizer = tl.TextLocalizer()
    lines_pos = localizer.findLines(imgGray, 1, ['size', 'borders'])
    lines_neg = localizer.findLines(imgGray, -1, ['size', 'borders'])

    croppedRegions = []
    for line in lines_pos:
      croppedRegions.append(TextCropper.getCroppedRegions(img, line))
    for line in lines_neg:
      croppedRegions.append(TextCropper.getCroppedRegions(img, line))
    return croppedRegions

  @staticmethod
  def getCroppedRegions(img, line):
    ((minY, minX), (maxY, maxX)) = line.bounds()
    return img[minY:maxY, minX:maxX]
    