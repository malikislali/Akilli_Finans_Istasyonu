from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard_kokpit, name='dashboard'), # Ana sayfa açıldığında direkt tetiklenir
    path('api/sinyal/<str:sembol>/', views.sinyal_api_endpoint, name='sinyal_api'), # Masa hatırlatıcının dinleyeceği hat
]