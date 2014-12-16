import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../', 'swt'))
import swt as swt
import textCropper as tc
import numpy as np
from matplotlib import pyplot as plt
import cv2
import math, random
from profiler import *
import itertools

def test_text_cropper():
  img = cv2.imread('test/images/stopsign.jpg', 0)
  croppedRegions = tc.TextCropper.cropTextRegionsFromImage(img)
  for i in range(len(croppedRegions)):
    plt.subplot(len(croppedRegions), 1, i+1)
    plt.imshow(croppedRegions[i])
  plt.show()

if __name__ == "__main__":
  test_text_cropper()