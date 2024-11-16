from django.urls import path
from app import views
urlpatterns=[
    path("",views.index,name="index"),
    path('process-image/', views.process_image, name='process_image'),
    path('get-result/', views.get_result, name='get_result')
]
