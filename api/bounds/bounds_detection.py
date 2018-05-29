import cv2 as cv
import numpy as np
import imutils


def boundsDetection(image):
    rectKernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 3))
    sqKernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.bitwise_not(gray)
    tophat = cv.morphologyEx(gray, cv.MORPH_TOPHAT, rectKernel)

    gradX = cv.Sobel(tophat, ddepth=cv.CV_32F, dx=1, dy=0,
                     ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype("uint8")

    gradX = cv.morphologyEx(gradX, cv.MORPH_CLOSE, rectKernel)
    thresh = cv.threshold(gradX, 0, 255,
                          cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, sqKernel)

    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
                           cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    boundsCoords = []

    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv.boundingRect(c)
        ar = w / float(h)
        if (w > 5) & (h > 5) & (ar > 1):
            boundsCoords.append([x, y, w, h])

    return boundsCoords
