# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 18:20:00 2020

@author: Lenovo
"""

from rest_framework import serializers
from .models import Faces


class SongsSerializer(serializers.ModelSerializer):
    class Meta:
        model = Faces
        fields = ("name", "date", "photo", "result_photo")
