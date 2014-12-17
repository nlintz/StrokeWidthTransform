import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../', 'lib'))
import lib.textCropper as tc
import numpy as np
from matplotlib import pyplot as plt
import cv2
import math, random
from profiler import *
import itertools

def test_text_cropper():
  img = cv2.imread('test/images/fallout_shelter.jpg')
  croppedRegions = tc.TextCropper.cropTextRegionsFromImage(img)
  for i in range(len(croppedRegions)):
    plt.subplot(len(croppedRegions), 1, i+1)
    b,g,r = cv2.split(croppedRegions[i])
    rgbImg = cv2.merge([r,g,b])
    cv2.imwrite('fallout_'+str(i)+'.jpg', croppedRegions[i])
    plt.imshow(rgbImg)
  plt.show()

if __name__ == "__main__":
  test_text_cropper()