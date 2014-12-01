import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../', 'swt'))
import swt
import numpy as np
from matplotlib import pyplot as plt
import cv2
import math
import ImageFilter

def test_castRay():
  img = np.zeros((100,100),dtype=np.uint8)
  img[:,0:4] = 255
  img[:,97:] = 255
  edges = cv2.Canny(img, 100.0, 200.0)
  angles = swt.gradient(edges)
  rayValid = swt.castRay(edges, angles, (0,3), antigradient=False)
  rayInvalid = swt.castRay(edges, angles, (0,3), antigradient=True)
  assert rayValid == [(0, i, 93.0) for i in range(3, 97)]
  assert swt.rayLength(rayValid) == 93.0
  assert rayInvalid == None

def test_castRayAtAngle():
  img = np.zeros((100,100),dtype=np.uint8)
  cv2.line(img, (20,0), (0,20), 255, 5)
  cv2.line(img, (90,99), (99,90), 255, 5)
  # cv2.line(img, (0,10), (99,10),255,1)

  edges = cv2.Canny(img, 100.0, 200.0)
  angles = swt.gradient(img)
  plt.imshow(edges, cmap="gray", interpolation="none")
  plt.show()


def test_imageSWT():
  img = cv2.imread('N.jpg',0)
  plt.imshow(swt.strokeWidthTransform(img), cmap="gray", interpolation="none")
  plt.show()

def test_gradient():
  # img = cv2.imread('diag.jpg',0)
  width = 500
  height = 500
  img = np.zeros((height,width), dtype=np.uint8)
  # cv2.line(img, (0,height-1), (height-1,width/2), (255, 255, 255), 3)
  # cv2.line(img, (0,height-1), (int(height-1/2.0),0), (255, 255, 255), 3)
  cv2.line(img, (0,0), (int((height-1)/2.0),height-1), (255, 255, 255), 3)

  grad = swt.gradient(img)
  print grad[:,30] * (360/(math.pi * 2))

def test_gen_lines():
  img = np.zeros((500,500,3), dtype=np.uint8)
  cv2.line(img, (0,50), (500,50), (255, 255, 255), 5)
  cv2.imwrite('horz.jpg', img)
  
  img = np.zeros((100,100, 3), dtype=np.uint8)
  cv2.line(img, (50,0), (50,99), (255, 255, 255), 5)
  cv2.imwrite('vert.jpg', img)
  
  img = np.zeros((100,100, 3), dtype=np.uint8)
  cv2.line(img, (0,0), (99,99), (255, 255, 255), 5)
  cv2.imwrite('diag.jpg', img)

def test_swt_sobel():
  img = cv2.imread('traffic.jpg',0)
  swt.strokeWidthTransform(img)

if __name__ == "__main__":
  test_swt_sobel()