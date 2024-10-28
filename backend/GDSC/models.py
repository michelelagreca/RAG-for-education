from django.db import models

# Create your models here.

class Item(models.Model):
    age = models.IntegerField(default=18)
    schoolOrJob = models.TextField(default="")
    studyDescription = models.TextField(default="")
    methodPreference = models.TextField(default="")
    studyGoal = models.TextField(default="")

    def __str__(self):
        return self.age

class BasicText(models.Model):
    text = models.TextField(default="")

    def __str__(self):
        return self.text
    