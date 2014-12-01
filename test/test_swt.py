import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../', 'swt'))
import swt
import numpy as np
from matplotlib import pyplot as plt
import cv2
import math


def test_imageSWT():
  img = cv2.imread('traffic.jpg',0)
  B,G,R = cv2.split(cv2.imread('traffic.jpg',1))
  img_color = cv2.merge((R,G,B))

  plt.subplot(1,3,1)
  plt.imshow(img_color, interpolation="none")
  plt.subplot(1,3,2)
  plt.imshow(swt.strokeWidthTransform(img, 1), cmap="gray", interpolation="none")
  plt.subplot(1,3,3)
  plt.imshow(swt.strokeWidthTransform(img, -1), cmap="gray", interpolation="none")

  plt.show()

def test_gradient():
  width = 500
  height = 500
  img = np.zeros((height,width), dtype=np.uint8)
  # cv2.line(img, (0,height-1), (height-1,width/2), (255, 255, 255), 3)
  # cv2.line(img, (0,height-1), (int(height-1/2.0),0), (255, 255, 255), 3)
  cv2.line(img, (0,0), (int((height-1)/2.0),height-1), (255, 255, 255), 3)

  grad = swt.gradient(img)
  print grad[:,30] * (360/(math.pi * 2))

if __name__ == "__main__":
  test_imageSWT()