# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.shortcuts import render
from django.http import HttpResponse
from django.http import HttpResponseRedirect
import pandas as pd
import numpy as np, numpy
import tensorflow as tf
from sklearn.externals import joblib
import os
import datetime
from datetime import date 
import math
from django.shortcuts import redirect
import scipy as sc
reloadMod=joblib.load(os.path.dirname(__file__)+"/aimodels/pulseoxy_test.pkl")
def symp(request,pid):
    pid=pid
    sp=0
    pulse=0
    temp=0
    res=0
    dias=0
    sys=0
    pdata=patient_data.objects.filter(id=pid)
    spo2=heartrate.objects.filter(pid__exact=pid,oxygen_saturation__gt=10).order_by('-date')[:1]
    hrate=heartrate.objects.filter(pid__exact=pid,pulse__gt=10).order_by('-date')[:1]
    resp=heartrate.objects.filter(pid__exact=pid,respiration__gt=10).order_by('-date')[:1]
    temper=heartrate.objects.filter(pid__exact=pid,temperature__gt=5).order_by('-date')[:1]
    diastolic=heartrate.objects.filter(pid__exact=pid,bps__gt=1).order_by('-date')[:1]
    for hrate in hrate:
        pulse=hrate.pulse
    for spo2 in spo2:
        sp=spo2.oxygen_saturation
    for resp in resp:
        res=resp.respiration
    for temper in temper:
        temp=temper.temperature
    for diastolic in diastolic:
        dias=diastolic.bpd
        sys=diastolic.bps
    context={
        'id':pid,
        'pdata':pdata,
        'sp':sp,
        'pulse':pulse,
        'res':res,
        'temp':temp,
        'dias':dias,
        'sys':sys
    }
    return render(request,'lungs.html',context)
def predictMPG(request):
    dump={}
    if request.method == 'POST':
        temp={}
        paid=request.POST.get('submit')
        temp['Age']=request.POST.get('age')
        temp['gender']=request.POST.get('gender')
        temp['heartrate']=request.POST.get('HEARTRATE')
        temp['spo2']=request.POST.get('SP')
        temp['Temperature']=request.POST.get('TEMP')
        temp['Systolic Pressure']=request.POST.get('systolic')
        temp['Diastolic Pressure']=request.POST.get('diastolic')
        temp['Respiratory Rate']=request.POST.get('RESPRATE')
        now=datetime.datetime.now()
        test=meter_checker(pid=paid, gender= temp['gender'], age=temp['Age'], meter_type='Lungs', heart_rate=temp['heartrate'],
        	temperature= temp['Temperature'], spo2=temp['spo2'], systolic=temp['Systolic Pressure'],
        	diastolic=temp['Diastolic Pressure'],respiration=temp['Respiratory Rate'],submit_date=now)
        test.save()
        dump=temp.copy()
        if temp['gender']=="Male":
        	dump['M']=1
        	del dump["gender"]
        else:
        	dump['M']=0
        	del dump["gender"]
    testdata=pd.DataFrame({'x':dump}).transpose()
    testdata= testdata[["Age","Diastolic Pressure","Systolic Pressure","Respiratory Rate","heartrate","Temperature","spo2","M"]]
    graph=tf.Graph()
    with graph.as_default():
        scoreval=reloadMod.predict(testdata)
    if scoreval>0.5:
        finalval="Lungs not in good condition"
    else:
        finalval="Lungs in good condition"
    result=meter_checker.objects.filter(pid__exact=paid).order_by('-submit_date')[:1]
    for result in result:
        meterid=result.mid
    cont=meter_result(mid=meterid,result=scoreval[0],description=finalval,result_date=now)
    pdata=patient_data.objects.filter(id=paid)
    cont.save()
    pdata=patient_data.objects.filter(id=paid)
    meter=meter_result.objects.filter(mid=meterid)[:1]
    mete=meter_checker.objects.filter(pid__exact=paid,meter_type__exact='Lungs').values('spo2').order_by('-submit_date')[:1]
    aivalue=scoreval[0]*100
    context={
        'paid':paid,
        'pdata':pdata,
        'aivalue':aivalue,
        'age':temp['Age'],
        'gender':temp['gender'],
        'meter':meter,
        'mete':mete
    }
    return render(request,'pulseoxy.html',context)