import random
import numpy as np
import math

from skimage.draw import line, line_aa, circle, set_color, circle_perimeter_aa
from skimage.io import imsave
from skimage.util import random_noise

maxSlope = 10 # restrict the maximum slope of generated lines for stability
minLength = 20 # restrict the minimum length of line segments

class CircleDataset:
	'''
	Generator of line segment images.

	Images will have 1 random line segment each, filled with noise and distractor circles.
	Class also offers functionality for drawing line parameters, hypotheses and point predictions.
	'''

	def __init__(self, imgW = 64, imgH = 64, margin = -5, bg_clr = 0.5):
		'''
		Constructor. 

		imgW -- image width (default 64)
		imgH -- image height (default 64)
		margin -- lines segments are sampled within this margin, negative value means that a line segment can start or end outside the image (default -5)
		bg_clr -- background intensity (default 0.5)
		'''
		
		self.imgW = imgW
		self.imgH = imgH
		self.margin = margin
		self.bg_clr = bg_clr

	def draw_circle(self, data, cX, cY, r, clr, alpha=1.0):
		'''
		Draw a circle with the given color and opacity.

		data -- image to draw to
		cX -- x value of circle center
		cY -- y value of circle center
		r -- radius of circle
		clr -- line color, triple of values
		alpha -- opacity (default 1.0)
		'''

		cY = int(cY * self.imgH)
		cX = int(cX * self.imgW)
		r = int(r * self.imgW)

		rr, cc, val =  circle_perimeter_aa(cY, cX, r)
		set_color(data, (rr, cc), clr, val)

	def draw_hyps(self, labels, scores, data=None):
		'''
		Draw a set of line hypothesis for a batch of images.
		
		labels -- line parameters, array shape (NxMx2) where 
			N is the number of images in the batch
			M is the number of hypotheses per image
			2 is the number of line parameters (intercept, slope)
		scores -- hypotheses scores, array shape (NxM), see above, higher score will be drawn with higher opacity
		data -- batch of images to draw to, if empty a new batch wil be created according to the shape of labels
		
		'''

		n = labels.shape[0] # number of images
		m = labels.shape[1] # number of hypotheses

		if data is None: # create new batch of images
			data = np.zeros((n, self.imgH, self.imgW, 3), dtype=np.float32)
			data.fill(self.bg_clr)

		clr = (0, 0, 1)

		for i in range (0, n):
			for j in range (0, m):
				lY1 = int(labels[i, j, 0] * self.imgH)
				lY2 = int(labels[i, j, 1] * self.imgW + labels[i, j, 0] * self.imgH)
				self.draw_line(data[i], 0, lY1, self.imgW, lY2, clr, scores[i, j])

		return data

	def draw_models(self, labels, data=None, correct=None):
		'''
		Draw circles for a batch of images.
	
		labels -- circle parameters, array shape (Nx3) where 
			N is the number of images in the batch
			3 is the number of circles parameters (center x,  center y, radius)
		data -- batch of images to draw to, if empty a new batch wil be created according to the shape of labels 
			and circles will be green, circles will be blue otherwise
		correct -- array of shape (N) indicating whether a circle estimate is correct 
		'''

		n = labels.shape[0]
		if data is None: 
			data = np.zeros((n, self.imgH, self.imgW, 3), dtype=np.float32)
			data.fill(self.bg_clr)
			clr = (0, 1, 0)
		else:
			clr = (0, 0, 1)

		for i in range (0, n):

			self.draw_circle(data[i], labels[i, 0], labels[i, 1], labels[i, 2], clr)

			if correct is not None:
				
				# draw border green if estiamte is correct, red otherwise
				if correct[i]: borderclr = (0, 1, 0)
				else: borderclr = (1, 0, 0)
				
				set_color(data[i], line(0, 0, 0, self.imgW-1), borderclr)			
				set_color(data[i], line(0, 0, self.imgH-1, 0), borderclr)			
				set_color(data[i], line(self.imgH-1, 0, self.imgH-1, self.imgW-1), borderclr)			
				set_color(data[i], line(0, self.imgW-1, self.imgH-1, self.imgW-1), borderclr)			

		return data

	def draw_points(self, points, data, inliers=None):
		'''
		Draw 2D points for a batch of images.

		points -- 2D points, array shape (Nx2xM) where 
			N is the number of images in the batch
			2 is the number of point dimensions (x, y)
			M is the number of points
		data -- batch of images to draw to
		inliers -- soft inlier score for each point, 
			if given and score < 0.5 point will be drawn green, red otherwise
		'''

		n = points.shape[0] # number of images
		m = points.shape[2] # number of points

		for i in range (0, n):
			for j in range(0, m):

				clr = (0.2, 0.2, 0.2) # draw predicted points as dark circles
				if inliers is not None and inliers[i, j] > 0.5:
					clr = (0.7, 0.7, 0.7) # draw inliers as light circles
					
				r = int(points[i, 0, j] * self.imgH)
				c = int(points[i, 1, j] * self.imgW)
				rr, cc = circle(r, c, 2)
				set_color(data[i], (rr, cc), clr)

		return data

	def sample_circles(self, n):
		'''
		Create new input images of random line segments and distractors along with ground truth parameters.

		n -- number of images to create
		'''

		data = np.zeros((n, self.imgH, self.imgW, 3), dtype=np.float32)
		data.fill(self.bg_clr)
		labels = np.zeros((n, 3), dtype=np.float32)
		
		for i in range (0, n): # for each image
		
			# -- STUDENT BEGIN --------------------------------------------------------

			# TASK 2.

			# You are given the tensors data and labels which you have to fill. 
			# Within this for loop, you work with data[i] and labels[i].

			# NOTE: remove the pass statement above this comment

			# data[i]:
			#	draw a random number of random line segments in random colors, e.g. 2 to 5 lines
			#	draw one circle with a random center position, 
			#		a random radius in the range of 0.1 * self.imgW to 0.5 * self.imgW
			#		and a random color
			#	see the line fitting code for reference

			# labels[i]:
			#	should contain x and y of the circle center, and radius of the circle in this order
			#	the labels should be in normalized image coordinates, i.e. divided by the image size


			# create a random number of distractor lines
			nC = random.randint(2, 5)
			for c in range(0, nC):
			
				while True:
					# sample segment end points
					lX1 = random.randint(self.margin, self.imgW-self.margin+1)
					lX2 = random.randint(self.margin, self.imgW-self.margin+1)
					lY1 = random.randint(self.margin, self.imgH-self.margin+1)
					lY2 = random.randint(self.margin, self.imgH-self.margin+1)

					# check min length
					length = math.sqrt((lX1 - lX2) * (lX1 - lX2) + (lY1 - lY2) * (lY1 - lY2))
					if length < minLength: continue

					# random color
					clr = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))

					# calculate line ground truth parameters
					delta = lX2 - lX1
					if delta == 0: delta = 1 

					slope = (lY2 - lY1) / delta
					intercept = lY1 - slope * lX1

					# not too steep for stability
					if abs(slope) < maxSlope: break
				
				rr, cc, val = line_aa(lX1, lY1, lX2, lY2)
				set_color(data[i], (rr, cc), clr, val)

			# create circle segment
			cR = random.randint(int(0.1 * self.imgW), int(1 * self.imgW))			
			cX1 = random.randint(int(-0.5 * cR), int(self.imgW+0.5*cR+1))
			cY1 = random.randint(int(-0.5 * cR), int(self.imgH+0.5*cR+1))	

			clr = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))

			labels[i, 0] = cX1 / self.imgW
			labels[i, 1] = cY1 / self.imgH
			labels[i, 2] = cR
			
			self.draw_circle(data[i], cX1, cY1, cR, clr)

			
			# -- STUDENT END ----------------------------------------------------------

			# apply some noise on top
			data[i] = random_noise(data[i], mode='speckle')

		return data, labels
