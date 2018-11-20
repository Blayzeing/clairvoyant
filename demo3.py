import PIL.Image
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.signal as sps
from functools import reduce
from scipy.ndimage.filters import gaussian_filter

SEARCH_RADIUS = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)] # The radius around each pixel to search for surrounding other lit blob parts

class Blob:
	# Each blob stores the weighted centre
	# [(x,y, weight)]
	def __init__(self, firstX, firstY, magnitude):
		self.points = [(firstX, firstY, magnitude)]
	def addPoint(self, x, y, magnitude):
		self.points.append((x,y,magnitude))
	def calculateCenterPointAndWeight(self):
		xSum = reduce(lambda total, x: total + x[0], self.points, 0)
		ySum = reduce(lambda total, y: total + y[1], self.points, 0)
		weightSum = reduce(lambda total, w: total + w[2], self.points, 0)
		pLen = len(self.points)
		self.xAvg = xSum/pLen
		self.yAvg = ySum/pLen
		self.weightAvg = weightSum/pLen
	def expand(self, image, thresh, mask):
		pointsLeft = list(self.points) # Stores a list of all locations checked
		while len(pointsLeft) > 0:
			currentPoint = pointsLeft[0]
			pointsLeft.remove(currentPoint)
			for s in SEARCH_RADIUS:
				# Get the value of the neighbour (swapping x/y to make it works with numpy)
				proposedPoint = (currentPoint[0]+s[0], currentPoint[1]+s[1])
				magnitudeAtPoint = image[proposedPoint[1], proposedPoint[0]]
				if (proposedPoint not in pointsLeft) and (proposedPoint not in mask) and (magnitudeAtPoint > thresh):
					pointsLeft.append(proposedPoint)
					self.points.append((proposedPoint[0], proposedPoint[1], magnitudeAtPoint))
					mask.append(proposedPoint)


# Functions
loadImage = lambda path: np.asarray(PIL.Image.open(path))
makeBadFilter = lambda w: np.asarray([(math.cos(((i/(w-1))*2-1)*math.pi)+1)/2 for i in range(w)])
def makeGaussianFilter(w):
	l = np.exp(-0.1*np.linspace(-1,1,w)**2)
	return l/np.trapz(l)
make2DFilter = lambda w, f: f(w)[:, np.newaxis] * f(w)[np.newaxis,:]
bound = lambda n, mx, mn: max(min(n, mx), mn)
def drawCross(image, position, size, thickness):
	if thickness < 1:
		thickness = 1
	thickness = int((thickness-1)/2)
	x = position[0]
	y = position[1]
	maxWidth = image.shape[1]-1
	maxHeight = image.shape[0]-1
	#for i in range(-size, size+1) for o in range(-thickness, thickness+1):
	for i, o in [(i, o) for i in range(-size, size+1) for o in range(-thickness, thickness+1)]:
		image[int(bound(y+i, maxHeight, 0)), int(bound(x+o, maxWidth, 0))] = 255
		image[int(bound(y+o, maxHeight, 0)), int(bound(x+i, maxWidth, 0))] = 255

# Variables
THRESHOLD_SCALE = 0.75 # How far from darkest to lightest change in photos that the threshold will be chosen for detecting lights


# TODO:
# 1. Smooth image
# 2. Identify LED locations
# 	2.1. Blur image
#	2.2. binarize > 75% or something (This is actually done in 2.3, on-the-fly, so the data's still there for the weighted average)
#	2.3. search for blobs by expanding out, creating mask of checked pixels - each blob is an object in a list, with a size and central mass (weighted average point)
#	2.4. Calculate average size and s.d., remove all non-1-sigma blobs
#	2.5. The remaining blobs are the LEDs.
#	2.X. WATCH OUT THAT THE AVERAGE SIZE AND S.D. OF THE BLOBS IS NOT TOO SMALL (Perhaps add a weighting function over size)
# 3. Estimate line positions using hoz/vert clustering/highest-point convolution
# 4. Split 2D points into 4 clusters based on which line they are closest to
# 5. Repeat [3,4] using different rotaitons, trying to calculate best rotation (could also use vertical/hoz height, as a cube will always be smaller when straight)
# 6. Estimate dewarp parameters by optimising to regression lines to each cluster
# 7. Scale until box is largest size/there is minimal downsampling at the center of the image


# Load image as array
image = loadImage("imgs/initial-tests-leds/1-all.jpg")

# Flatten image to single channel
flattened = np.max(image, axis=2)
filterSize = int((flattened.shape[0]/2.5))
FILTER_SIGMA = 7


#plt.imshow(gaussian_filter(flattened, sigma=FILTER_SIGMA))
#plt.show()

b = Blob(10,20,0.5)
print(b)
b.addPoint(10,10, 2)
print(b)
b.calculateCenterPointAndWeight()
print(b)
print(b.xAvg)
print(b.yAvg)
print(b.weightAvg)

# Calculate the range of shades, and get the median shad
threshold = (np.max(flattened) - np.min(flattened)) * THRESHOLD_SCALE + np.min(flattened)
print(threshold)

# Loop over every pixel and assemble blobs.
blobs = []
mask = np.zeros((flattened.shape[0], flattened.shape[1])) # Stores which points have been converted to blobs.
mask = []
for y in range(0,flattened.shape[0]):
	for x in range(0,flattened.shape[1]):
		if (flattened[y,x] > threshold):
			if ((x,y) not in mask):
				print("found new blob (%i)"%(len(blobs)+1))
				newBlob = Blob(x,y,flattened[y,x])
				newBlob.expand(flattened, threshold, mask)
				blobs.append(newBlob)
				mask.append((x,y))

print("%i blobs detected."%(len(blobs)))
print("Calculating blob centerpoints...")
for b in blobs:
	b.calculateCenterPointAndWeight()
print("Centerpoints calculated.")
print([(b.xAvg, b.yAvg) for b in blobs])
for pos in [(b.xAvg, b.yAvg) for b in blobs]:
	drawCross(flattened, pos, 20, 3)
plt.imshow(flattened)
plt.show()


