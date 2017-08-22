from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^classify', views.classify, name='classify'),
    url(r'^similarity', views.similarity, name='similarity'),
]