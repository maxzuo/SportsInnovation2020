import cv2
import numpy as np

WEIGHTS_FILE = "cfg/yolov3.weights"
CONFIG_FILE = "cfg/yolov3.cfg"
NAMES_FILE = "cfg/coco.names"

class YOLO(object):

	def __init__(self, w_file=WEIGHTS_FILE, c_file=CONFIG_FILE, n_file=NAMES_FILE):
		# initialize model
		self.net = cv2.dnn.readNet(w_file, c_file)

		# read classes
		with open(c_file, 'r') as f: self.classes = [line.strip() for line in f.readlines()]

		layer_names = self.net.getLayerNames()
		self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
		
		# colors = np.random.uniform(0, 255, size=(len(classes), 3)) # probably not necesary

	def predict(self, img):
		original_shape = img.shape
		# print(img.shape)
		blob = cv2.dnn.blobFromImage(img, 0.00392, (608, 608), (0, 0, 0), True, crop=False)
		# print(blob.shape)
		self.net.setInput(blob)
		outs = self.net.forward(self.output_layers)
		# print(outs)


if __name__ == "__main__":
	img = cv2.imread("img/test.png")

	yolo_net = YOLO()
	yolo_net.predict(img)