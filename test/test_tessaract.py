import Image
import pytesseract
import cv2
import numpy as np

cv2_images = [cv2.imread('images/fallout_6.jpg',0)]

for image in cv2_images:
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	cl_image = clahe.apply(image)
	cl_image = np.uint8(cl_image*255)
	pil_im = Image.fromarray(cl_image)
	print pytesseract.image_to_string(pil_im, lang="eng")
	print "*******"



#im = Image.fromarray(np.uint8(cm.gist_earth(myarray)*255))


img = cv2.imread('images/fallout_6.jpg',0)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)
cv2.imwrite('images/fallout_6_better.jpg', cl1)
cv2_im = cv2.imread('images/fallout_6_better.jpg',0)
pil_im = Image.fromarray(cv2_im)
print pytesseract.image_to_string(pil_im, lang="eng")



"""
print ("******")
print pytesseract.image_to_string(Image.open('images/fallout_5_better.jpg'), lang="eng")
print ("******")
print pytesseract.image_to_string(Image.open('images/fallout_6_better.jpg'), lang="eng")
"""