import PIL.Image
import numpy as np
import math
import matplotlib.pyplot as plt

# Load image as array
loadImage = lambda path: np.asarray(PIL.Image.open(path))
makeBadFilter = lambda w: [(math.cos(((i/(w-1))*2-1)*math.pi)+1)/2 for i in range(w)]


image = loadImage("imgs\\initial-tests-leds\\1-all.jpg")

# Flatten image
flattened = np.max(image, axis=2)

# Making histograms
hoz = np.sum(flattened, axis=0)
vert = np.sum(flattened, axis=1)


filterSize = flattened.shape[0]/2.5

def drawVerticalLineAt(point, data):
	dividerLine = [0 if i<point else data.max() for i in range(data.shape[0])]
	plt.plot(dividerLine)
def drawHorizontalLineAt(point, data):
	line = [point for i in range(data.shape[0])]
	plt.plot(line)
def convolveAndShow(data, filterS):
	convolved = np.convolve(data, makeBadFilter(int(filterS)), mode="same")
	avg = np.mean(convolved)

	plt.plot(convolved)
	drawHorizontalLineAt(avg, convolved)
	plt.show()


def guesstimateCenter(data, filterS):
	convolved = np.convolve(data, makeBadFilter(int(filterS)), mode="same")
	weightDistributions = convolved/convolved.sum() # normalize the points into bins so that they can be used to find a weighted average to find the point between the two sides
	avgSeparationPoint = np.mean(weightDistributions * np.arange(0,convolved.shape[0])) * convolved.shape[0]
	print(avgSeparationPoint)

	plt.plot(convolved)
	drawVerticalLineAt(avgSeparationPoint, convolved)
	plt.show()
	return int(avgSeparationPoint)



def guestimateLinePeaks(data, filterS, centerMark):
	convolved = np.convolve(data, makeBadFilter(int(filterS)), mode="same")
	points = (convolved[:centerMark].argmax(), convolved[centerMark:].argmax() + centerMark)
	plt.plot(convolved)
	drawVerticalLineAt(points[0], convolved)
	drawVerticalLineAt(points[1], convolved)
	plt.show()
	return points

#convolveAndShow(hoz, filterSize)
#convolveAndShow(vert, filterSize)

hozCenter = guesstimateCenter(hoz, filterSize)
vertCenter = guesstimateCenter(vert, filterSize)

hozEdges = guestimateLinePeaks(hoz, filterSize, hozCenter)
vertEdges = guestimateLinePeaks(vert, filterSize, vertCenter)

print (hozEdges, vertEdges)

# TODO:
# Build pose scoring function, probably should just sum all of the total heights of each edge curve.
# Attempt rotation to optimise pose scoring first
# Then attempt dewarping
