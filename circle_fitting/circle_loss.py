import torch
import numpy as np

class CircleLoss:
	'''
	Compares two circle by distance of parameter values.
	'''

	def __init__(self, image_size):
		'''
		Constructor.

		image_size -- size of the input images, used to normalize the loss
		'''
		self.image_size = image_size

	def __call__(self, est, gt):
		'''
		Calculate the circle loss.

		est -- estimated circle, form: [cX, cY, r]
		gt -- ground truth circle, form: [cX, cY, r]
		'''
		
		# -- STUDENT BEGIN --------------------------------------------------------

		# TASK 3.

		# You are given the estimated and ground truth circle (both given as 3-vectors containting 
		# the center coordinate and radius.

		# Calculate the loss as the Euclidean distance between the circle centers plus the absolute difference in radii.
		# Note that the given circle parameters are in relative coordinate, i.e. from 0 to 1. The loss should be in pixels,
		# so multiply by self.image_size

		# You should return one value: the loss.

		# -- STUDENT END ----------------------------------------------------------
		dist_circle_center = torch.dist(torch.tensor([est[0], est[1]]), torch.tensor([gt[0], gt[1]]), p=2) * self.image_size
		diff_radii = torch.abs(gt[2] - est[2]) * self.image_size
		return dist_circle_center + diff_radii
