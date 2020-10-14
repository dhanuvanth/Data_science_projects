from django.conf.urls import url
from corona import views
urlpatterns = [
    url(r'^symptoms/(?P<pid>\d+)/$',views.symp),
	url(r'^predictMPG',views.predictMPG, name='PredictMPG'),
	
]