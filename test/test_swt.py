import sys, os
sys.path.insert(0, '../')
import lib.swt as swt
import numpy as np
from matplotlib import pyplot as plt
import cv2
import math
from lib.profiler import *

def test_profileSWT():
  imgs = ['036.jpg', 'billboard-cropped.jpg',
   'billboard.jpg', 'sofsign.jpg', 'traffic.jpg']

  for imgname in imgs:
    img = cv2.imread('images/'+imgname,0)
    print 'Timing: ' + imgname
    swt_pos = swt.strokeWidthTransform(img, 1)

def test_imageSWT():
  filename = 'images/rab_butler_large.JPG'
  img = cv2.imread(filename,0)
  B,G,R = cv2.split(cv2.imread(filename,1))
  img_color = cv2.merge((R,G,B))
  swt_pos = swt.strokeWidthTransform(img, 1)
  swt_pos_dilated = 255 - cv2.dilate(255 - swt_pos, kernel = np.ones((2,2),np.uint8), iterations = 2)
  swt_neg = swt.strokeWidthTransform(img, -1)
  swt_neg_dilated = 255 - cv2.dilate(255 - swt_neg, kernel = np.ones((2,2),np.uint8), iterations = 2)

  plt.subplot(3,2,1)
  plt.imshow(img_color, interpolation="none")
  plt.title('original image')

  plt.subplot(3,2,3)
  plt.imshow(swt_pos, cmap="gray", interpolation="none")
  plt.title('positive swt of image')
  cv2.imwrite('results/rab_butler/swt_pos.jpg', swt_pos)
  plt.subplot(3,2,4)
  plt.imshow(swt_pos_dilated, cmap="gray", interpolation="none")
  plt.title('dilated positive swt of image')

  plt.subplot(3,2,5)
  plt.title('negative swt of image')
  plt.imshow(swt_neg, cmap="gray", interpolation="none")
  cv2.imwrite('results/rab_butler/swt_neg.jpg', swt_neg)
  plt.subplot(3,2,6)
  plt.title('dilated negative swt of image')
  plt.imshow(swt_neg_dilated, cmap="gray", interpolation="none")

  plt.show()

def test_edge_detect():
  filename = 'images/elevator.jpg'
  img = cv2.imread(filename,0)
  th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
  edges = cv2.Canny(th, 100, 200)
  plt.imshow(edges, 'gray')
  plt.show()

def test_gradient():  
  filename = 'images/rab_butler.jpg'
  img = cv2.imread(filename,0)
  edges = cv2.Canny(img, 100, 300)
  thetas = swt.gradient(img, edges)
  plt.imshow(thetas, 'gray')
  plt.show()

def test_first_pass():
  filename = 'images/elevator.jpg'
  img = cv2.imread(filename,0)
  edges = swt.adaptiveEdges(img)
  thetas = swt.gradient(img, edges)
  firstPass, rays = fastRay.castRays(edges, thetas, 1)
  plt.imshow(firstPass, 'gray')
  plt.show()

def test_plot_rays():
  filename = 'images/elevator.jpg'
  img = cv2.imread(filename,0)
  edges = cv2.Canny(img, 100, 300)
  thetas = swt.gradient(img, edges)
  firstPass, rays = fastRay.castRays(edges, thetas, -1)

  rayPlot = np.zeros((img.shape[0], img.shape[1]))
  for ray in rays:
    for pixel in ray:
      rayPlot[pixel[0], pixel[1]] = 255
  plt.imshow(rayPlot, 'gray')
  plt.show()

def test_first_and_second_pass():
  filename = 'images/elevator.jpg'
  img = cv2.imread(filename,0)
  edges = cv2.Canny(img, 100, 300)
  thetas = swt.gradient(img, edges)
  firstPass, rays = fastRay.castRays(edges, thetas, 1)
  secondPass = swt.refineRays(firstPass, rays)
  plt.imshow(secondPass, 'gray')
  plt.show()

if __name__ == "__main__":
  test_imageSWT()