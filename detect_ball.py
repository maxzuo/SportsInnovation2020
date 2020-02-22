from model import *
from time import sleep
from model import YOLO

WEIGHTS_FILE = "cfg/yolov3.weights"
CONFIG_FILE = "cfg/yolov3.cfg"
NAMES_FILE = "cfg/coco.names"

stream = cv2.VideoCapture('./videos/1904GATCOFFvsPATE.mp4') ## needs to be changed

def find_ball(img):
    model = YOLO(w_file = WEIGHTS_FILE, c_file = CONFIG_FILE, n_file=NAMES_FILE)
    boxes = model.predict(img, 32)
    return boxes


def count_balls(stream):
    count = 0
    while True:
        ret, frame = stream.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = 128
        gray = cv2.cvtColor(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)

        if find_ball(grey) != None:
            count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    return count


def ballFinding_yolo(path):
    img = cv2.imread(path)

    boxes = find_ball(img)

    if boxes == None:
        print("Ball not found by YOLO.")
    else:
        for i in range(len(boxes)):
                x, y, w, h = new_boxes[i]
                print(x, y, w, h)
                label = "Ball%d" % i
                color = np.array(self.colors[0])
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

        cv2.imwrite("ballFound.png", img)


def ballFinding_contours(streamPath):
    stream = cv2.VideoCapture(streamPath)  ## streamPath should be a video clip of the main arch of the pass

    prevFrame = None
    count = 0
    while count <= 10:  ## check 10 frames
        count += 1
        ret, frame = stream.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = 128
        img = cv2.cvtColor(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)

        if cv2.waitKey(1) & 0xFF == ord('q'):   ## needed if using imshow
            break

        if count == 1:
            prevFrame = img
            continue

        delta = img - prevFrame
        thresh = cv2.threshold(delta, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        cv2.imwrite("./contours/delta" + str(count) + ".png", delta)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        image = cv2.drawContours(frame, contours, -1, (0,255,0), 3)
        prevFrame = img

        cv2.imwrite("./contours/contour" + str(count) + ".png", image)


def detect_flag(streamPath):
    stream = cv2.VideoCapture(streamPath)

    while True:
        ret, frame = stream.read()

        lowerY = np.array([240, 240, 0])
        upperY = np.array([255, 255, 0])
        mask = cv2.inRange(frame, lowerY, upperY)

        img = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow("Flag Detection", img)
        cv2.imshow("Original Frames", frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


streamPath = "./videos/penaltyClip.mp4"
detect_flag(streamPath)