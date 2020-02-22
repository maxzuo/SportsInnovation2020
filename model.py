import cv2
import numpy as np
from sklearn import svm
import math
import matplotlib.pyplot as plt
from datetime import datetime

WEIGHTS_FILE = "cfg/yolov3.weights"
CONFIG_FILE = "cfg/yolov3.cfg"
NAMES_FILE = "cfg/coco.names"

class YOLO(object):

	def __init__(self, confidence=0.01, w_file=WEIGHTS_FILE, c_file=CONFIG_FILE, n_file=NAMES_FILE):
		# initialize model
		self.net = cv2.dnn.readNet(w_file, c_file)
		self.confidence = confidence

		# read classes
		with open(n_file, 'r') as f: self.classes = [line.strip() for line in f.readlines()]
		# print(self.classes)

		layer_names = self.net.getLayerNames()
		self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
		
		self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3)) # probably not necesary

	def predict(self, img, p_class=0):
		original = img
		height, width, channels = original_shape = img.shape
		#try HSV: did not work
		

		# 0.00392 is 1/255.0
		blob = cv2.dnn.blobFromImage(original, 0.00392, (608, 608), (0, 0, 0), True, crop=False)
		# print(blob.shape)
		self.net.setInput(blob)
		outs = self.net.forward(self.output_layers)
		# print(outs)

		confidences = []
		boxes = []
		# class_ids = []

		for out in outs:
			for detection in out:
				# print(detection)
				scores = detection[4:]
				class_id = np.argmax(scores)
				if class_id != p_class:
					continue
				# class_ids.add(class_id)

				confidence = scores[class_id]
				if confidence > self.confidence:
					center_x = int(detection[0] * width)
					center_y = int(detection[1] * height)
					w = int(detection[2] * width)
					h = int(detection[3] * height)

					x = int(center_x - w / 2)
					y = int(center_y - h / 2)
					boxes.append([x, y, w, h])
					confidences.append(float(confidence))
					# class_ids.append(class_id)
		# print(boxes)
		# print(len(boxes))

		indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.3).flatten()
		# print(indexes)
		# index_set = set()
		# for i in indexes: index_set.add(i)

		areas = []
		new_boxes = []

		# for i in range(len(boxes)):
		# 	if i in index_set:
			# 	_, _, w, h = boxes[i]
			# 	areas.append(np.sqrt(w * h))
			# 	new_boxes.append(boxes[i])

		for i in indexes:
			_, _, w, h = boxes[i]
			areas.append(w * h)
			new_boxes.append(boxes[i])
		
		average_area = np.median(areas)
		#print(average_area)

		del boxes
		boxes = []

		font = cv2.FONT_HERSHEY_PLAIN
		# consider using filter instead of this method
		for i in range(len(new_boxes)):
			if ((areas[i] < 0.2 * average_area) or (areas[i] > 4 * average_area)):
				continue
			boxes.append(new_boxes[i])

			# commented out because affecting images
			# x, y, w, h = new_boxes[i]
			# print(x, y, w, h)
			# label = "Person%d" % i
			# color = np.array(self.colors[0])
			# cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
			# cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

		# cv2.imwrite("result.png", img)
		return np.array(boxes)

def mask(img, return_mask=False):
	lower_green = np.array([40,40, 40])
	upper_green = np.array([150, 255, 255])

	mask = cv2.inRange(img, lower_green, upper_green)

	kernel = np.ones((5, 5))
	mask = cv2.erode(mask, kernel, iterations=5)
	mask = cv2.dilate(mask, kernel, iterations=7)

	if return_mask:
		return mask

	img = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
	
	return img

def kmeans(img, k=3, return_labels=False, restarts = 15):

	if len(img.shape) == 3:
		img = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
	
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
	flags = cv2.KMEANS_RANDOM_CENTERS

	_, labels, centroids = cv2.kmeans(np.float32(img), k, None, criteria, restarts, flags)

	quantized = np.bincount(labels.flatten())

	if return_labels:
		return labels
	return centroids[quantized.argsort()[::-1]]


def assign_teams(img, boxes, masked=None):
	start = datetime.now()
	if masked is None:
		img = mask(img)
	else:
		img = masked
	print("Mask:", datetime.now() - start)
	start = datetime.now()
	colors = [None] * len(boxes)
	minX = 10_000
	minY = 10_000
	maxX = 0
	maxY = 0

	

	for i in range(len(boxes)):
		x, y, w, h = boxes[i]
		minX = min(minX, x + w/2)
		minY = min(minY, y + y/2)
		maxX = max(maxX, x + w/2)
		maxY = max(maxY, y + h/2)

	maxX -= minX
	maxY -= minY

	X = np.zeros((len(boxes,)))
	Y = np.zeros((len(boxes,)))
	positions = [None] * len(boxes)
	primaries = []
	secondaries = []

	for i in range(len(boxes)):
		x, y, w, h = boxes[i]
		# img[y:y+h,x:x+w]
		cv2.imwrite("img/%d.png" % i,img[y:y+h,x+int(0.4*w):x+int(0.6*w)])
		print(x+w//2, x+w//2 + 2)
		color = kmeans(np.float32(cv2.resize(img[y:y+h,x+w//2:x+w//2 + 2], None,fx=1.0, fy=1.0), restarts=10))
		
		# line below for higher accuracy
		# color = kmeans(np.float32(img[y:y+h,x:x+w]), restarts=10))
		color = color.tolist()
		color.remove(min(color, key=lambda c: np.sum(c)))
		x = x + w/2
		y = y + h/2

		x = ((x - minX) / maxX - 0.5) * 2
		y = ((y - minY) / maxY - 0.5) * 2

		X[i] = x
		Y[i] = y
		positions[i] = [x, y]
		colors[i] = np.concatenate([np.asarray(color[0]) / 255,np.asarray(color[1]) / 255, np.asarray([0, 0])])
		primaries.append(color[0])
		secondaries.append(color[1])
	
	cv2.imwrite("sorted_colors.png", np.array([np.repeat(np.array(primaries), 20, axis=0).flatten()]).repeat(20, axis=0).reshape(20, -1, 3))
	secondaries.sort(key=lambda a: np.sum(a))
	cv2.imwrite("sorted_colors_1.png", np.array([np.repeat(np.array(secondaries), 20, axis=0).flatten()]).repeat(20, axis=0).reshape(20, -1, 3))
	
	
	# print(colors)
	# input()

	primary_colors = np.asarray(colors)
	assignments = kmeans(primary_colors, k=2, restarts=10, return_labels=True)
	# print(np.bincount(assignments.flatten()))
	# input()
	# res = [None] * len(boxes)
	return X, Y, np.array(positions), assignments.flatten()
	
def get_line(X, labels):
	# teams should have 0 or 1 labels, not colors, at this stage

	svc = svm.SVC(kernel='linear', C=0.001).fit(X, labels)
	W = svc.coef_[0]
	I = svc.intercept_
	slope = -W[0]/W[1]
	b = -I[0]/W[1]
	return slope, I

def transform_coordinates(positions, slope, intercept):
	newCoords = []
	theta = math.atan(slope)
	transformation = np.array([[np.cos(-theta), -np.sin(theta)], [np.sin(theta), np.cos(-theta)]])
	return transformation.dot(positions.T), intercept * math.cos(-theta)


if __name__ == "__main__":
	img = cv2.imread("img/test.png")

	yolo_net = YOLO()
	
	start = datetime.now()	
	boxes = yolo_net.predict(img)
	print("YOLO finished", datetime.now() - start)
	start = datetime.now()

	# valueMask = np.array(np.array([0, 0, 255], dtype=np.uint8).reshape((1, 1, 3)))
	# valueMask = valueMask.resize(img.shape[:-1])
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	cv2.imwrite("original_hsv.png", hsv)
	# print(hsv)
	hsv[:, :, 2] = 255
	
	cv2.imwrite("flattened_hsv.png", hsv)
	newImg = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

	cv2.imwrite("flattened.png", newImg)

	masked = mask(img, return_mask=True)
	newImg = cv2.bitwise_and(newImg, newImg, mask=cv2.bitwise_not(masked))
	cv2.imwrite("newImg.png", newImg)

	X, Y, positions, labels = assign_teams(newImg, boxes)
	print("Assigned teams", datetime.now() - start)
	start = datetime.now()

	slope, intercept = get_line(positions, labels)
	rotated, intercept = transform_coordinates(positions, slope, intercept)
	print(intercept)
	print("rotated", datetime.now() - start)
	start = datetime.now()
	
	


	# PLOT RESULTS
	plt.scatter(X, Y, c=labels)
	# plt.plot(X, [intercept + slope * i for i in X], "b-", label="Line of Scrimmage")
	# plt.show()
	# ax.plot(list(range(1, maxK + 1)), kNN_euc_loss, "g-", label="Testing Loss")
	
	plt.savefig("not_rotated.png")
	plt.clf()

	X = rotated[:][0]
	Y = rotated[:][1]
	# print(X.reshape(-1, 1).shape, colors.T[:][:3].T.shape)
	# assignments = kmeans(np.concatenate((colors.T[:][:3].T, Y.reshape(-1, 1) * 1.5), axis=1), k=2, restarts=20, return_labels=True)
	# print(assignments.flatten() - labels)
	plt.scatter(X, Y, c=labels.flatten())
	plt.plot(X, np.ones(X.shape) * intercept, "b-", label="Line of Scrimmage")
	plt.savefig("rotated_1.png")
	plt.clf()

	

	# print(teams[0])
	# font = cv2.FONT_HERSHEY_PLAIN
	# color = yolo_net.colors[0]
	# for position, assignment in teams:
	# 	x, y = position
	# 	label = str(assignment)
	# 	cv2.putText(img, label, (int(x), int(y) + 30), font, 3, color, 3)
	cv2.imwrite("labels.png", img)

	
























































	# for _, color in teams:
	# print([color[0] for _, color in teams])
	# for i in range(1):
	# 	primary_colors = np.asarray([(np.array(color[i]) / 255).tolist() +[position[0] / 1000, position[1] / 1000] for position, color in teams])
	# 	# print(primary_colors.shape)
	# 	assignments = kmeans(primary_colors, k=2, return_labels=True)
	# 	print(assignments)
		# print(np.bincount(kmeans(primary_colors, k=2, return_labels=True).flatten()))
		# summation = np.sum(primary_colors, axis=1)
		# # print(summation)
		# primary_colors = primary_colors[summation.argsort()]
# (np.array(color[i]) / 255).tolist() +

		# cv2.imwrite("sorted_colors_%d.png" % i, np.array([np.repeat(np.array(primary_colors), 20, axis=0).flatten()]).repeat(20, axis=0).reshape(20, -1, 3))

	# for a in primary_colors:
		# print(np.sum(a), a)
	

	