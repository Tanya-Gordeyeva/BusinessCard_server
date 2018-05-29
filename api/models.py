from django.db import models


class Names(models.Model):
    name = models.TextField(primary_key=True)


class Lastnames(models.Model):
    lastname = models.TextField(primary_key=True)


class Fathernames(models.Model):
    fathername = models.TextField(primary_key=True)
