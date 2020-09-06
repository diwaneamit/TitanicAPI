#Author: Amit Diwane
#Description: This is an API for Model Consumption
from django.shortcuts import render
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response 
from rest_framework import status
from django.http import JsonResponse
from django.core import serializers
from django.conf import settings
import json
import joblib


# Create your views here.
@api_view(["POST"])
def get_titanic(data):
    try:
        #read Import Parameter
        iv_data = json.loads(data.body)
        PassengerId = iv_data["PassengerId"] 	
        Age = iv_data["Age"]
        SibSp = iv_data["SibSp"] 	
        Parch = iv_data["Parch"]	
        male_gender = iv_data["male_gender"] 	
        C = iv_data["C"]
        Q = iv_data["Q"] 	
        S = iv_data["S"]
        
        #Create Model Reference Variable
        cls = joblib.load("Titanic_model_consume.sav")
        
        #Append list with data
        list = []
        list.append(PassengerId)
        list.append(Age)
        list.append(SibSp)
        list.append(Parch)
        list.append(male_gender)
        list.append(C)
        list.append(Q)
        list.append(S)
        
        #Call to predict value from model and return it
        ans = cls.predict([list])
        return JsonResponse('You will' + str(ans),safe=False)
    except ValueError as err:
        return Response(err.args[0],status.HTTP_400_BAD_REQUEST)          
