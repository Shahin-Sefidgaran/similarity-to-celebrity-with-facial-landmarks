from django.urls import path

from . import views

app_name = 'face'
urlpatterns = [
    path('upload/', views.FileUploadView.as_view()),
    path('index/', views.index, name='index'),
    path('result/', views.send, name='send'),
    path('train/', views.FileTrainView.as_view(), name='train'),
    path('face/', views.ListFacesView.as_view(), name="faces-all")
]