from model import YOLO
import cv2
import os
import tensorflow as tf
from collections import deque
import numpy as np

from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Layer
from tensorflow.keras.optimizers import Adam

data_folder = "training_images/"
save_folder = "training_images/boxes"
X = []
Y = []

def box_gen(folder, images):
    model = YOLO()
    for path in images:
        print(path)
        path = os.path.join(data_folder, path)
        img = cv2.imread(path)
        
        boxes = model.predict(img)
        yield img, boxes

def mse(a, b):
    return float(((a-b) ** 2).mean().flatten())

def save_img(img, orientation, rep=False):
    if orientation == "left":
        rev = "right"
        y = np.array([1, 0, 0, 0]) # left, right, up, down
    elif orientation == "right":
        rev = "left"
        y = np.array([0, 1, 0, 0])
    X.append(cv2.resize(img, (100, 100)))
    Y.append(y)
    assert len(X) == len(Y)
    if len(X) % 10 == 0:
        np.save(os.path.join(save_folder, "X.npy"), np.asarray(X))
        np.save(os.path.join(save_folder, "Y.npy"), np.asarray(Y))
        print("SAVED", len(X))
    if not rep:
        save_img(cv2.flip(img, 1), rev, rep=True)

def process_images(folder, images):
    gen = box_gen(folder, images)
    prev_tags = []
    prev_tags = deque(prev_tags, 500)
    for img, boxes in gen:
        # tags = []
        print("LENGTH", len(prev_tags))
        if len(boxes) < 10:
            continue
        for x, y, w, h in boxes:
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            try:
                roi = img[y:y+h,x:x+w]
            except:
                continue
            # if roi.shape[0] == 0 or roi.shape[1] == 0:
            #     print(x, y, w, h)
            #     continue
            # print(x, y)
            # for p, t in prev_tags:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi = cv2.resize(roi, (500, 500))
            if prev_tags:
                min_mse = -1
                min_p = -1
                min_t = None
                for p, t in prev_tags:
                    score = mse(p, roi)
                    if min_mse == -1 or score < min_mse:
                        min_mse = score
                        min_p = p
                        min_t = t
                # print(min_mse, min_t)
                if min_mse < 30 and min_mse != -1:
                    
                    save_img(roi, min_t)
                    continue
                # min([mse(p, roi) for p, t in prev_tags])
            cv2.imshow("roi", roi)
            key = cv2.waitKeyEx(0)
            if key == 2424832:
                prev_tags.append((roi, "left"))
                save_img(roi, "left")
            if key == 2555904:
                prev_tags.append((roi, "right"))
                save_img(roi, "right")
                # print("right")
            if key == ord('q'):
                break
        # del prev_tags
        # prev_tags = tags

        

if __name__ == "__main__":
    images = list(filter(lambda p: os.path.isfile(os.path.join(data_folder, p)), os.listdir(data_folder)))
    images.sort(key=lambda name: int(name[:-4]))
    process_images(data_folder, images)