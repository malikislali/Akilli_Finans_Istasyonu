"""
quant_web_project/urls.py DEĞİŞİKLİĞİ
========================================
Mevcut dosyan şu an:

    from django.contrib import admin
    from django.urls import path, include

    urlpatterns = [
        path('admin/', admin.site.urls),
        path('', include('quant_app.urls')),
    ]

Aşağıdaki TEK satırı ekle (web route'larına dokunmadan):

    from django.contrib import admin
    from django.urls import path, include

    urlpatterns = [
        path('admin/', admin.site.urls),
        path('api/v1/', include('quant_app.api_urls')),   # 🆕 mobil app
        path('', include('quant_app.urls')),                # web (aynen duruyor)
    ]
"""
