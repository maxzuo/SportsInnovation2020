try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import cv2
import numpy as np
import os
import time


def detect_text(path):
    """Detects text in the file."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print('Texts:')

    for text in texts:
        print('\n"{}"'.format(text.description))

        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in text.bounding_poly.vertices])

        print('bounds: {}'.format(','.join(vertices)))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))




edge = False     # set to True for edge detection, set to False for thresholding

outFile = open("OCR_Results.txt", 'w')

directory = './scoreboards'
images = [cv2.imread('scoreboards/' + filename) for filename in sorted(os.listdir(directory), key=lambda p: int(p[:-4]))]
imgCount = 0
for image in images:

    if imgCount == 0:
        image = image[660:740]
    else:
        image = image[595:740, :]
    
    if edge:
        grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(grayImg, 50, 150, apertureSize=3)
        cv2.imwrite("OCRedges.png", edges)
        image = edges
    else:
        image = cv2.inRange(image, np.array([150, 150, 150]), np.array([255, 255, 255]))

    cv2.imwrite("ocrTEST" + str(imgCount) + ".png", image)

    qtrImg = image[:, 100:450]
    cv2.imwrite("ocrSegmentResults/QTRocrTest" + ".png", qtrImg)
    timeImg = image[:, 480:800]
    cv2.imwrite("ocrSegmentResults/TIMEocrTest" + ".png", timeImg)
    downPosImg = image[:, 1650:-100]
    cv2.imwrite("ocrSegmentResults/DOWN_POSocrTest" + ".png", downPosImg)

    print("Image " + str(imgCount))
    imgCount += 1
    

    #str(imgCount) + 


    #cv2.imwrite("ocrTestImages/ocrTest" + str(imgCount) + ".png", image)

    """
    ## We only need to include S, T, Q, R, A, N, D, H, L, O, B
    custom_config = r'-c tessedit_char_blacklist=abcdefghijklmnopqrstuvwxyzCEFIJKMOPUVWXYZ!@#%&(){}<>/-_;~` --psm 6'
    output = pytesseract.image_to_string(Image.open('ocrTestImages/ocrTest'+ str(imgCount)  +'.png'), config=custom_config)

    if "BALL" in output:
        print("Location: " + output[output.find("BALL ON") : output.find("BALL ON") + 10])
    if ":" in output:
        print("Time: " + output[output.find(":") - 2: output.find(":") + 3])
    if "QTR" in output:
        print("Quarter: " + output[output.find("QTR") - 4: output.find("QTR") + 2])
    """

    vars = [qtrImg, timeImg, downPosImg]
    for i in range(len(vars)):
        if i == 0:
            # Include S, T, N, D, R, Q, H
            custom_config = r'-c tessedit_char_blacklist=056789abcdefghijklmnopqrstuvwxyzABCEFGIJKLMOPUVWXYZ!:@#%&(){}<>/-_;~` --psm 6'
            output = pytesseract.image_to_string(Image.open('ocrSegmentResults/QTRocrTest.png'), config=custom_config)
            if "1ST" in output or "2ND" in output or "3RD" in output or "4TH" in output or "QTR" in output and '\n' not in list(output):
                outFile.write(str(imgCount) + '-  ' + output + '  ')
            print(str(imgCount) + ": " + output)
        elif i == 1:
            custom_config = r'-c tessedit_char_blacklist=abcdefghijklmnopqrstuvwxyzABCEFGHIJKLMOPUVWXYZ!@#%&(){}<>/-_;~` --psm 6'
            output = pytesseract.image_to_string(Image.open('ocrSegmentResults/TIMEocrTest.png'), config=custom_config)
            if ":" in output and '\n' not in list(output):
                outFile.write(output + '  ')
        elif i == 2:
            # Include A, B, D, N, L, O, R, S, T, H
            custom_config = r'-c tessedit_char_blacklist=abcdefghijklmnopqrstuvwxyzCEFGIJKMPUVWXYZ!@#%&():{}<>/-_;~` --psm 6'
            output = pytesseract.image_to_string(Image.open('ocrSegmentResults/DOWN_POSocrTest.png'), config=custom_config)
            if "BALL" in output or "ON" in output or "AND" in output or "1ST" in output or "2ND" in output or "3RD" in output or "4TH" in output and '\n' not in list(output):
                outFile.write(output + '\n')





# French text image to string


# Get bounding box estimates
#print(pytesseract.image_to_boxes(Image.open('ocr.png')))

# Get verbose data including boxes, confidences, line and page numbers
#print(pytesseract.image_to_data(Image.open('ocr.png')))

# Get information about orientation and script detection
#print(pytesseract.image_to_osd(Image.open('ocr.png')))

# Get a searchable PDF
#pdf = pytesseract.image_to_pdf_or_hocr('ocr.png', extension='pdf')
#with open('test.pdf', 'w+b') as f:
#    f.write(pdf) # pdf type is bytes by default

# Get HOCR output
#hocr = pytesseract.image_to_pdf_or_hocr('ocr.png', extension='hocr')
#print(hocr)
