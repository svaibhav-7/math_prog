from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('graphical/', views.graphical, name='graphical'),
    path('simplex/', views.simplex, name='simplex'),
]
