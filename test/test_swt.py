import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../', 'swt'))
import swt
import numpy as np
from matplotlib import pyplot as plt
import cv2
import math

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
  img = np.zeros((100,100), dtype=np.uint8)
  # cv2.rectangle(img, (40,30), (60,80), 255, -1)
  cv2.line(img, (0,0), (99,99), 255, 2)
  # cv2.line(img, (50,0), (50,99), 255, 5)
  # cv2.line(img, (0,50), (99,50), 0, 5)
  # angles = swt.gradient(img)
  # print angles[:, 40]
  plt.imshow(img, cmap="gray", interpolation="none")
  plt.show()

def test_canny():
  img = cv2.imread('N.jpg',0)
  img = (img)
  edges = cv2.Canny(img, 100, 200)
  sobelx = cv2.Sobel(edges, cv2.CV_32F, 1, 0, ksize = 5, scale = -1)
  sobely = cv2.Sobel(edges, cv2.CV_32F, 0, 1, ksize = 5, scale = -1)
  angles = np.arctan2(sobely, sobelx)
  # angles = cv2.phase(sobelx, sobely)
  # angles = swt.gradient(edges)
  print angles[50,:]*360/(math.pi*2)
  plt.imshow(cv2.phase(sobelx, sobely), cmap="gray", interpolation="none")
  # plt.plot(angles)
  plt.show()


if __name__ == "__main__":
  test_canny()