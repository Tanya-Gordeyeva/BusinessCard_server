from django.contrib import admin
from django.urls import path, include
from . import operations

urlpatterns = {
    path('qr', operations.qr),
    path('bounds', operations.bounds),
    path('textRecognition', operations.textRecognition),
    path('classification', operations.textClassification),
    path('train',operations.trainData)
}
