""" 
Python Connected Components Implementation
Developed by Diana Vermilya and Nathan Lintz
"""

import cv2
import Queue
import numpy as np
import math

def generateListOfAllPixels(rows, cols):
	all_pixels = []
	for i in range(rows):
		for j in range(cols):
			all_pixels.append((i,j))
	return all_pixels

def thresholded_cc():
	swt = cv2.imread('sample_swt.png',0)
	rows = swt.shape[0]
	cols = swt.shape[1]
	new_image = np.zeros((rows, cols,3), np.uint8)

	q = Queue.Queue()
	enqueued = {}
	tags = {}
	tag_count = 0
	all_pixels = generateListOfAllPixels(rows, cols)
	for i in range(len(all_pixels)):
		first_pix  = all_pixels[i]
		if not first_pix in enqueued:
			tags[tag_count] = []
			q.put(first_pix)
			enqueued[first_pix] = True

			while not q.empty():	
				[y,x] = q.get()
				b_shade = swt[y,x]*(-1) + 255 + 0.001

				for pix in [(y,x-1), (y,x+1), (y-1,x), (y+1,x)]:
					if pix[0] >= 0 and pix[0] < rows and pix[1] >= 0 and pix[1] < cols:
						n_shade = swt[pix[0], pix[1]]*(-1) + 255 + 0.001
						if float(n_shade)/b_shade < 3 and float(n_shade)/b_shade > 0.33:
							if not pix in enqueued:
								q.put(pix)
								enqueued[pix] = True
				tags[tag_count].append((y,x,swt[y,x]))
				new_image[y,x] = (i*15)
			tag_count += 1

	cv2.imwrite('blank.png', new_image)
	return tags

def filter_by_size(regions):
	SIZE_TOLERANCE = 10
	return {k:v for (k,v) in regions.iteritems() if len(v) > SIZE_TOLERANCE}

def meets_variance_tolerance(region):
	VARIANCE_TOLERANCE = 2.0
	mean_width = sum([w for (y,x,w) in region])/float(len(region))

	for (y,x,w) in region:
		if w < mean_width/VARIANCE_TOLERANCE:
			return False
	return True

def filter_by_variance(regions):
	return {k:v for (k,v) in regions.iteritems() if meets_variance_tolerance(v) == True}

def meets_aspect_ratio_and_diameter(region, bounds):
	(min_y, max_y, min_x, max_x) = bounds
	y_diff = max_y - min_y
	x_diff = float(max_x - min_x + 0.001)
	aspect_ratio = y_diff/x_diff

	weights = [w for (y,x,w) in region]
	median_weight = np.median(np.array(weights))
	diameter = math.sqrt(y_diff**2 + x_diff**2)

	if aspect_ratio < 10 and aspect_ratio > 0.1:
		if diameter/median_weight < 10:
			return True
	return False

def filter_by_aspect_ratio_and_diameter(regions, bounds_map):
	return {k:v for (k,v) in regions.iteritems() if meets_aspect_ratio_and_diameter(v, bounds_map[k]) == True}


def map_to_bounds(regions):
	bounds = {}
	for (key, region) in regions.iteritems():
		min_y = min([y for (y,x,w) in region])
		max_y = max([y for (y,x,w) in region])
		min_x = min([x for (y,x,w) in region])
		max_x = max([x for (y,x,w) in region])
		bounds_for_region = (min_y, max_y, min_x, max_x)
		bounds[key] = bounds_for_region
	print bounds
	return bounds

def contains(bounds_a, bounds_b):
	(min_ya, max_ya, min_xa, max_xa) = bounds_a
	(min_yb, max_yb, min_xb, max_xb) = bounds_b
	if min_ya < min_yb and max_ya > max_yb and min_xa < min_xb and max_xa > max_xb:
		return True

def filter_out_borders(regions, bounds):
	#400 gets filtered out here
	regions_contained = {}
	keys = regions.keys()
	for region in keys:
		for other_region in keys:
			if contains(bounds[region], bounds[other_region]):
				regions_contained[region] = regions_contained.get(region, 0) + 1
			regions_contained[region] = regions_contained.get(region, 0)

	print regions_contained
	return {k:v for (k,v) in regions.iteritems() if regions_contained[k] < 3}





regions = thresholded_cc()
bounds_map = map_to_bounds(regions)


print len(regions)
regions = filter_by_size(regions)
print len(regions)
regions = filter_by_variance(regions)
print len(regions)
regions = filter_by_aspect_ratio_and_diameter(regions, bounds_map)
print len(regions)
regions = filter_out_borders(regions, bounds_map)
print len(regions)



