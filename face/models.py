from django.db import models
from persiantools.jdatetime import JalaliDateTime

# Create your models here.

class Faces(models.Model):
    name = models.CharField(max_length=255)
    date = models.CharField(max_length=10, default=JalaliDateTime.today().strftime("%Y/%m/%d"), editable=False)
    photo = models.FileField(upload_to='media', blank=True)
    result_photo = models.CharField(max_length=500)
    
    def __str__(self):
        return "{} - {}".format(self.name, self.date)