from rest_framework import serializers
from .models import Item, Example

class ItemSerializer(serializers.ModelSerializer):
    class Meta:
        model = Item
        fields = ('age', 'schoolOrJob', 'studyDescription', 'methodPreference', 'studyGoal')

class BasicTextSerializer(serializers.ModelSerializer):
    class Meta:
        model = BasicText
        fields = ('text')