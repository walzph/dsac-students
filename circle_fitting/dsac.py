import torch
import torch.nn.functional as F

import random
import numpy as np
import math

class DSAC:
	'''
	Differentiable RANSAC to robustly fit lines.
	'''

	def __init__(self, hyps, inlier_thresh, inlier_beta, inlier_alpha, loss_function):
		'''
		Constructor.

		hyps -- number of line hypotheses sampled for each image
		inlier_thresh -- threshold used in the soft inlier count, its measured in relative image size (1 = image width)
		inlier_beta -- scaling factor within the sigmoid of the soft inlier count
		inlier_alpha -- scaling factor for the soft inlier scores (controls the peakiness of the hypothesis distribution)
		loss_function -- function to compute the quality of estimated line parameters wrt ground truth
		'''

		self.hyps = hyps
		self.inlier_thresh = inlier_thresh
		self.inlier_beta = inlier_beta
		self.inlier_alpha = inlier_alpha
		self.loss_function = loss_function

	def define_circle(self, p1, p2, p3):
		"""
		Returns the center and radius of the circle passing the given 3 points.
		"""
		a = torch.det(torch.tensor([
			[p1[0], p1[1], 1],
			[p2[0], p2[1], 1],
			[p3[0], p3[1], 1]
		]))
		b = torch.det(-torch.tensor([
			[p1[0] ** 2 + p1[1] ** 2, p1[1], 1],
			[p2[0] ** 2 + p2[1] ** 2, p2[1], 1],
			[p3[0] ** 2 + p3[1] ** 2, p3[1], 1]
		]))
		c = torch.det(torch.tensor([
			[p1[0] ** 2 + p1[1] ** 2, p1[0], 1],
			[p2[0] ** 2 + p2[1] ** 2, p2[0], 1],
			[p3[0] ** 2 + p3[1] ** 2, p3[0], 1]
		]))
		d = torch.det(-torch.tensor([
			[p1[0] ** 2 + p1[1] ** 2, p1[0], p1[1]],
			[p2[0] ** 2 + p2[1] ** 2, p2[0], p2[1]],
			[p3[0] ** 2 + p3[1] ** 2, p3[0], p3[1]]
		]))

		""" if torch.abs(a) < 1.0e-6 or torch.abs(b) < 1.0e-6 or torch.abs(c) < 1.0e-6 or torch.abs(d) < 1.0e-6:
			return None, None, None """

		# Center of circle
		cx = -b / (2 * a)
		cy = -c / (2 * a)

		radius = torch.sqrt((b ** 2 + c ** 2 - 4 * a * d) / (4 * a ** 2) + 0.00001)
		return cx, cy, radius


	def __sample_hyp(self, x, y):
		'''
		Calculate a circle hypothesis (cX, cY, r) from three random points.

		x -- vector of x values (range 0 to 1)
		y -- vector of y values (range 0 to 1)
		'''

		# -- STUDENT BEGIN --------------------------------------------------------

		# TASK 4.1.

		# You are given all predicted points of the CNN (vector of x coordinates, and vector of y coordinates)
		# Randomly select three points (but no point twice within this call) and fit a circle 
		# (see the slides for instructions).

		# Check if the radius is below 1 (sanity check). If not, try again. Try 1000 times max.
		
		# You should return four values: x of the circle center, y of the circle center, the radius,
		# and True resp. False if no valid circle has been found within 1000 tries (radius < 1).

		# -- STUDENT END ----------------------------------------------------------
		x = x.clone().detach().numpy()
		y = y.clone().detach().numpy()

		centerX, centerY, radius = 0,0,0

		found = False
		for sanityCheck in range(0, 1000):

			#select three random points
			randomSlice = np.random.choice(len(x), size=3, replace=False)
			randX = x[randomSlice]
			randY = y[randomSlice]
			
			""" 
			A = randX[0] * (randY[1] - randY[2]) - randY[0] * (randX[1] - randX[2]) + randX[1]*randY[2] - randX[2] * randY[1]
			B = (randX[0] ** 2 + randY[0] ** 2) * (randY[2] - randY[1]) + (randX[1] ** 2 + randY[1] ** 2) * (randY[0] - randY[2]) + (randX[2] ** 2 + randY [2] ** 2) * (randY[1] - randY[0])
			C = (randX[0] ** 2 + randY[0] ** 2) * (randX[1] - randX[2]) + (randX[1] ** 2 + randY[1] ** 2) * (randX[2] - randX[0]) + (randX[2] ** 2 + randY [2] ** 2) * (randX[0] - randX[1])
			D = (randX[0] ** 2 + randY[0] ** 2) * (randX[2]*randY[1] - randX[1]*randY[2]) + (randX[1] ** 2 + randY[1] ** 2) * (randX[0]*randY[2] - randX[2]* randY[0]) + (randX[2] ** 2 + randY [2] ** 2) * (randX[1]*randY[0] - randX[0]*randY[1])
			centerX = - B / (2 * A)
			centerY = - C / (2 * A)
			radius = np.sqrt( ((B**2) + (C**2) - 4*A*D) / 4 * (A**2))
 			"""
			centerX, centerY, radius = self.define_circle(torch.tensor([randX[0], randY[0]]), torch.tensor([randX[1], randY[1]]), torch.tensor([randX[2], randY[2]]))

			#sanity check
			if radius < 1.0:
				found = True
				break
		return centerX, centerY, radius, found

	def __soft_inlier_count(self, cX, cY, r, x, y):
		'''
		Soft inlier count for a given circle and a given set of points.

		cX -- x of circle center
		cY -- y of circle center
		r -- radius of the circle
		x -- vector of x values (range 0 to 1)
		y -- vector of y values (range 0 to 1)
		'''

		# -- STUDENT BEGIN --------------------------------------------------------

		# TASK 4.2.

		# You are given the circle parameters cX, cY and r, as well as all predicted points of the CNN (x, y)
		# Calculate the distance of each point to the circle.
		# Turn the distances to soft inlier scores by applying a sigmoid as in the line fitting code.
		# Use the member attributes self.inlier_beta as the scaling factor within the sigmoid,
		# and self.inlier_thresh as the soft inlier threshold (see line fitting code).

		# Note that when using the sqrt() function, add an epsilon to the argument since the gradient of sqrt(0) is unstable.

		# You should return two values: a score for the circle (sum of soft inlier scores), 
		# and a vector with the soft inlier score of each point.

		# -- STUDENT END ----------------------------------------------------------
		dist = torch.abs(torch.sqrt( (x - cX)**2 + (y - cY)**2  + 0.0000001 ) - r)
		#inlinerCount = torch.nonzero(torch.where(dist < self.inlier_thresh, dist, torch.zeros(dist.shape))).size()[0]

		dist = 1 - torch.sigmoid(self.inlier_beta * (dist - self.inlier_thresh)) 
		score = torch.sum(dist)

		return score, dist

	def __refine_hyp(self, x, y, weights):
		'''
		Refinement by least squares fit.

		x -- vector of x values (range 0 to 1)
		y -- vector of y values (range 0 to 1)
		weights -- vector of weights (1 per point)		
		'''

		# -- STUDENT BEGIN --------------------------------------------------------

		# TASK 4.3. (and 4.4.)

		# You are given all predicted points of the CNN (x, y) and a soft inlier weight for each point (weights).
		# Do a least squares fit to all points with weight > 0.5, or a weighted least squares fit to all points.
		# A description can be found in materials/circle_fit.pdf

		# Note that PyTorch offers a differentiable inverse() function. 

		# You should return three values: x of the circle center, y of the circle center, and the radius

		# -- STUDENT END ----------------------------------------------------------

		#least square with all points > 0.5
		xSub = x[weights > 0.5]
		ySub = y[weights > 0.5]

		if xSub.shape[0] == 0 or ySub.shape[0] == 0:
			return 0,0,0

		xMean = torch.mean(xSub)
		yMean = torch.mean(ySub)
		#print("mean", xMean, yMean)
		u = xSub - xMean
		v = ySub - yMean

		N = u.shape[0]

		#calc S for Eq. 3 and 4
		Suu = torch.sum(u * u)
		Suv = torch.sum(u * v)
		Svv = torch.sum(v * v)
		Suuu = torch.sum(u * u * u)
		Svvv = torch.sum(v * v * v)
		Suvv = torch.sum(u * v * v)
		Svuu = torch.sum(v * u * u)
		
		#print(Suu, Suv, Svv, Suuu, Svvv, Suvv, Svuu)

		#Equation solving
		leftSide = np.array([[Suu, Suv], [Suv, Svv]])
		rightSide = np.array([0.5 * (Suuu + Suvv), 0.5* (Svvv + Svuu)])
		#print("left, right", leftSide, rightSide)

		try:
			solution = np.linalg.solve(leftSide, rightSide)
		except:
			return 0,0,0
		#print("u,v", solution)
		centerX = solution[0] + xMean
		centerY = solution[1] + yMean
		
		#Eq 6
		radius =  np.sqrt(solution[0]**2 + solution[1]**2 + (Suu + Svv) / N + 0.000001)

		return centerX, centerY, radius
		

	def __call__(self, prediction, labels):
		'''
		Perform robust, differentiable line fitting according to DSAC.

		Returns the expected loss of choosing a good line hypothesis which can be used for backprob.

		prediction -- predicted 2D points for a batch of images, array of shape (Bx2) where
			B is the number of images in the batch
			2 is the number of point dimensions (y, x)
		labels -- ground truth labels for the batch, array of shape (Bx2) where
			B is the number of images in the batch
			2 is the number of parameters (intercept, slope)
		'''

		# working on CPU because of many, small matrices
		prediction = prediction.cpu()

		batch_size = prediction.size(0)

		avg_exp_loss = 0 # expected loss
		avg_top_loss = 0 # loss of best hypothesis

		self.est_parameters = torch.zeros(batch_size, 3) # estimated lines
		self.est_losses = torch.zeros(batch_size) # loss of estimated lines
		self.batch_inliers = torch.zeros(batch_size, prediction.size(2)) # (soft) inliers for estimated lines

		for b in range(0, batch_size):

			hyp_losses = torch.zeros([self.hyps, 1]) # loss of each hypothesis
			hyp_scores = torch.zeros([self.hyps, 1]) # score of each hypothesis

			max_score = 0 	# score of best hypothesis

			y = prediction[b, 0] # all y-values of the prediction
			x = prediction[b, 1] # all x.values of the prediction

			for h in range(0, self.hyps):	

				# === step 1: sample hypothesis ===========================
				cX, cY, r, valid = self.__sample_hyp(x, y)
				if not valid: continue

				# === step 2: score hypothesis using soft inlier count ====
				score, inliers = self.__soft_inlier_count(cX, cY, r, x, y)

				# === step 3: refine hypothesis ===========================
				cX_ref, cY_ref, r_ref = self.__refine_hyp(x, y, inliers)

				if r_ref > 0: # check whether refinement was implemented
					cX, cY, r = cX_ref, cY_ref, r_ref

				hyp = torch.zeros([3])
				hyp[0] = cX
				hyp[1] = cY
				hyp[2] = r

				# === step 4: calculate loss of hypothesis ================
				loss = self.loss_function(hyp, labels[b]) 

				# store results
				hyp_losses[h] = loss
				hyp_scores[h] = score

				# keep track of best hypothesis so far
				if score > max_score:
					max_score = score
					self.est_losses[b] = loss
					self.est_parameters[b] = hyp
					self.batch_inliers[b] = inliers

			# === step 5: calculate the expectation ===========================

			#softmax distribution from hypotheses scores			
			hyp_scores = F.softmax(self.inlier_alpha * hyp_scores, 0)

			# expectation of loss
			exp_loss = torch.sum(hyp_losses * hyp_scores)
			#print(exp_loss, hyp_losses, hyp_scores)
			avg_exp_loss = avg_exp_loss + exp_loss

			# loss of best hypothesis (for evaluation)
			avg_top_loss = avg_top_loss + self.est_losses[b]
		return avg_exp_loss / batch_size, avg_top_loss / batch_size
