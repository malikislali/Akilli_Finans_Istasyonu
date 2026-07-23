from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/v1/', include('quant_app.api_urls')),   # 🆕 mobil app
    path('', include('quant_app.urls')),                # web (aynen duruyor)
]