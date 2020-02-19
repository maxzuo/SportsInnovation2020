import cv2
import numpy as np
from sklearn import svm

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

	def predict(self, img):
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
				if class_id != 0:
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

		indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.3)

		areas = []
		new_boxes = []

		for i in range(len(boxes)):
			if i in indexes:
				_, _, w, h = boxes[i]
				areas.append(np.sqrt(w * h))
				new_boxes.append(boxes[i])
		
		average_area = np.median(areas)
		#print(average_area)

		del boxes
		boxes = []

		font = cv2.FONT_HERSHEY_PLAIN
		for i in range(len(new_boxes)):
			if ((areas[i] < 0.2 * average_area) or (areas[i] > 2 * average_area)):
				continue
			boxes.append(new_boxes[i])

			# commented out because affecting images
			# x, y, w, h = new_boxes[i]
			# print(x, y, w, h)
			# label = "Person%d" % i
			# color = np.array(self.colors[0])
			# cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
			# cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

		cv2.imwrite("result.png", img)
		return boxes

def mask(img):
	lower_green = np.array([40,40, 40])
	upper_green = np.array([100, 255, 255])

	mask = cv2.inRange(img, lower_green, upper_green)

	kernel = np.ones((5, 5))
	mask = cv2.erode(mask, kernel, iterations=5)
	mask = cv2.dilate(mask, kernel, iterations=7)
	img = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
	return img
"""
	k = 3

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
	flags = cv2.KMEANS_RANDOM_CENTERS
	_, labels, centroids = cv2.kmeans(pixels, k, None, criteria, 10, flags)

	palette = np.uint8(centroids)
	quantized = palette[labels.flatten()]
	quantized = quantized.reshape(fist.shape)

	#dominantColor = palette[np.argmax(itemfreq(labels)[:, -1])]

"""
def color_extraction(img):
	k = 5
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
	flags = cv2.KMEANS_RANDOM_CENTERS
	img = img.reshape(img.shape[0]*img.shape[1], 3)
	# print(img.shape)
	_, labels, centroids = cv2.kmeans(img, k, None, criteria, 10, flags)
	# print(centroids)
	# print(labels, labels.shape)

	quantized = np.bincount(labels.flatten())
	# quantized = quantized.reshape(img.shape)
	# print(quantized)
	# input()

	# print(centroids[quantized.bincount()])

	return centroids[quantized.argsort()[::-1]]



def assign_teams(img, boxes):
	img = mask(img)
	colors = [None] * len(boxes)
	for i in range(len(boxes)):
		x, y, w, h = boxes[i]
		cv2.imwrite("%d.png" % i, img[y:y+h,x:x+w])
		color = color_extraction(np.float32(img[y:y+h,x:x+w]))
		color = color.tolist()
		color.remove(min(color, key=lambda c: np.sum(c)))
		# color = list(filter(lambda c: np.sum(c) > 30, color))
		# assert len(color) == 2
		colors[i] = ((x + w/2, y + h/2), color)
	
	# print(colors)
	# input()
	
	return colors # for now
	
def get_line(teams):
	# teams should have 0 or 1 labels, not colors, at this stage
	X = []
	y = []
	for i in range(len(teams)):
		X.append(teams[i][0])
		y.append(teams[i][1])

	svc = svm.svc(kernel='linear').fit(np.array(X), np.array(y))
	W = svc.coef[0]
	I = svc.intercept_
	return (W, I)

def transform_coordinates(teams, a, b):
	pass


if __name__ == "__main__":
	img = cv2.imread("img/test.png")

	yolo_net = YOLO()
	
	
	boxes = yolo_net.predict(img)

	teams = assign_teams(img, boxes)
	
	# for _, color in teams:
	# print([color[0] for _, color in teams])
	for i in range(2):
		primary_colors = np.asarray([color[i] for _, color in teams])
		summation = np.sum(primary_colors, axis=1)
		# print(summation)
		primary_colors = primary_colors[summation.argsort()]


		cv2.imwrite("sorted_colors_%d.png" % i, np.array([np.repeat(np.array(primary_colors), 20, axis=0).flatten()]).repeat(20, axis=0).reshape(20, -1, 3))

	# for a in primary_colors:
		# print(np.sum(a), a)
	