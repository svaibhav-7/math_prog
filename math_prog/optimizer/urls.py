from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('graphical/', views.graphical, name='graphical'),
    path('simplex/', views.simplex_page, name='simplex'),
    path('transportation/', views.transportation_page, name='transportation_page'),
    path('transportation/', views.transportation_page, name='transportation_page'),
]
