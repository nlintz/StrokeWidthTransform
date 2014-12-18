import cv2
def bgr2rgb(img):
	b,g,r = cv2.split(img)       # get b,g,r
	return cv2.merge([r,g,b])     # switch it to rgb
