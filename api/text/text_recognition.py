import pytesseract
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def textRecognitionTesseract(img,bounds):
    (height, width, s) = img.shape
    r = 0
    g = 0
    b = 0
    pix = height * width
    for i in range(height):
        for j in range(width):
            r += img[i][j][0]
            g += img[i][j][1]
            b += img[i][j][2]
    colors = [r / pix, g / pix, b / pix]
    blur = cv.GaussianBlur(img, (5, 5), 0)
    if (colors[0] < 140) & (colors[1] < 140) & (colors[2] < 140):
        av = (colors[0]+colors[1]+colors[2])/3
        a, img = cv.threshold(blur, av+25, 255, cv.THRESH_BINARY_INV)
    else:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
        close = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel1)
        div = np.float32(gray) / (close)
        res = np.uint8(cv.normalize(div, div, 0, 255, cv.NORM_MINMAX))
        img = cv.cvtColor(res, cv.COLOR_GRAY2BGR)
        a, img = cv.threshold(img, 172, 255, cv.THRESH_BINARY)
    for i in range(height):
        for j in range(width):
            m = 0
            for k in range(len(bounds)):
                if (j > bounds[k][0] - 2) & (j < bounds[k][0] + bounds[k][2] + 2) & (i > bounds[k][1] - 2) & (
                        i < bounds[k][1] + bounds[k][3] + 2):
                    m = 1
                    break
            if m != 1:
                img[i][j] = [255, 255, 255]
    text = pytesseract.image_to_string(img, lang='rus+eng+deu')
    text_arr = text.split("\n")
    res = [value for value in text_arr if value and value != " "]
    result = "\n".join(res)
    return result
