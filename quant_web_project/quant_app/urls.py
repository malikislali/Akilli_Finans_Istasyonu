"""
quant_app/urls.py — App seviyesi route'lar.

🆕 NAVİGASYON YENİDEN YAPILANDIRILDI:
  '' (ana sayfa)   -> anasayfa_view (YENİ market overview ekranı)
  '/laboratuvar/'  -> dashboard_view (ESKİ tek-varlık ML analiz ekranı,
                       İÇERİK OLARAK DOKUNULMADI, sadece URL'i değişti)
  '/market/<pazar>/' -> market_view (pazara tıklanınca açılan ekran:
                       arama + periyot + İÇİNDE Laboratuvar'ın analiz
                       arayüzü)
"""
from django.urls import path
from . import views

urlpatterns = [
    path('', views.anasayfa_view, name='anasayfa'),
    path('laboratuvar/', views.dashboard_view, name='dashboard'),
    path('market/<str:pazar>/', views.market_view, name='market'),

    path('register/', views.register_view, name='register'),
    path('plan-sec/', views.plan_sec_view, name='plan_sec'),
    path('odeme/baslat/', views.iyzico_odeme_baslat, name='iyzico_odeme_baslat'),
    path('odeme/callback/', views.iyzico_callback, name='iyzico_callback'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),

    path('analiz/', views.analiz_api_view, name='analiz_api'),
    path('analiz/durum/<str:task_id>/', views.analiz_durum_view, name='analiz_durum'),

    # 🆕 Anasayfa / Market API'ları
    path('api/anasayfa-fiyatlar/', views.anasayfa_fiyatlar_api, name='anasayfa_fiyatlar_api'),
    path('api/market-fiyatlar/<str:pazar>/', views.market_fiyatlar_api, name='market_fiyatlar_api'),
    path('api/pazar-lider/<str:pazar>/', views.pazar_lider_api, name='pazar_lider_api'),
    path('api/tarama/<str:pazar>/', views.tarama_api, name='tarama_api'),
    path('api/tarama-gosterge/<str:pazar>/', views.tarama_gosterge_api, name='tarama_gosterge_api'),
    path('api/gosterge-serileri/', views.gosterge_serileri_api, name='gosterge_serileri_api'),

    # 🆕 Takip Listesi API'ları
    path('api/limit-durumu/', views.limit_durumu_api, name='limit_durumu_api'),
    path('api/usd-try-kuru/', views.usd_try_kuru_api, name='usd_try_kuru_api'),    path('api/takip-listesi/', views.takip_listesi_getir, name='takip_listesi_getir'),
    path('api/takip-listesi/ekle/', views.takip_listesi_ekle, name='takip_listesi_ekle'),
    path('api/takip-listesi/sil/<int:oge_id>/', views.takip_listesi_sil, name='takip_listesi_sil'),
]
