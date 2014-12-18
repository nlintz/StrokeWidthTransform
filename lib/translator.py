import numpy as np
import cv2
import pytesseract
import Image

class Translator(object):

	@staticmethod
	def find_text(img):
		""" translates an rgb image using tesseract
		"""
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
		cl1 = clahe.apply(gray)
		cv2.imwrite('placeholder.jpg', cl1)
		cv2_im = cv2.imread('placeholder.jpg',0)
		pil_im = Image.fromarray(cv2_im)
		return pytesseract.image_to_string(pil_im, lang="eng")
