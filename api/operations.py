from django.http import HttpResponse
from api.bounds.bounds_detection import boundsDetection
from api.text.text_recognition import textRecognitionTesseract
from api.classification.train import trainD
from api.classification.classificator import textClassificator

import pyzbar.pyzbar as pyzbar
import cv2
import numpy as np
from django.views.decorators.csrf import csrf_exempt
import json
from imageio import imread
import io
import base64


@csrf_exempt
def qr(request):
    if request.method == 'POST':
        im = request.read()
        nparr = np.fromstring(im, np.uint8)
        imDecoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        decodedObjects = pyzbar.decode(imDecoded)
    return HttpResponse(decodedObjects[0].data) if decodedObjects else HttpResponse([])


@csrf_exempt
def bounds(request):
    if request.method == 'POST':
        im = request.read()
        nparr = np.fromstring(im, np.uint8)
        imDecoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        bounds = boundsDetection(imDecoded)
        result = json.dumps(bounds)
    return HttpResponse(result)


@csrf_exempt
def textRecognition(request):
    if request.method == 'POST':
        im = request.body
        data_json = json.loads(im)
        bounds =  json.loads(data_json['bounds'])
        image = imread(io.BytesIO(base64.b64decode(data_json['image'])))
        imDecoded = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        text = textRecognitionTesseract(imDecoded, bounds)
    return HttpResponse(text)


@csrf_exempt
def textClassification(request):
    text = request.read().decode("utf-8")
    res = textClassificator(text)
    return HttpResponse(res) if res else HttpResponse("No trained data",status=422)


def trainData(request):
    result = trainD()
    return HttpResponse("OK") if result else HttpResponse("Bad request",status=422)
