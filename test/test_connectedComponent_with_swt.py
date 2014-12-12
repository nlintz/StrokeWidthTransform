import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../', 'swt'))
import swt.swt as swt
import swt.connected_components as cc
import numpy as np
from matplotlib import pyplot as plt
import cv2
import math
from profiler import *

@timeit
def test_cc_with_swt():
  img = cv2.imread('test/stopsign.jpg', 0)
  swt_pos = swt.strokeWidthTransform(img, 1)
  swt_pos_dilated = 255 - cv2.dilate(255 - swt_pos, kernel = np.ones((2,2),np.uint8), iterations = 2)
  regions = cc.connectComponents(swt_pos)
  ccImg = cc.connectedComponentsToImg(swt_pos, regions, img.shape[0], img.shape[1])

  plt.imshow(ccImg)
  plt.show()

if __name__ == "__main__":
  test_cc_with_swt()