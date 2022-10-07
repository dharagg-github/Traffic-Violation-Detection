from django.urls import path
from . import views

urlpatterns = [
    path("",views.index,name = "home"),
    path('detect',views.detect, name='detect'),
    path('einfo',views.einfo, name='einfo'),
]