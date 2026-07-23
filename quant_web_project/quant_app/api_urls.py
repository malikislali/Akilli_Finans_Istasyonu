"""
quant_app/api_urls.py — GÜNCELLENDİ (Adım 19)
Bu dosyanın TAMAMINI mevcut api_urls.py'nin yerine koy.
"""
from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView

from .api_views_auth import KayitView, GirisView, BenKimimView, CikisView
from .api_views_data import (
    WatchlistView, WatchlistSilView,
    AlarmlarView, AlarmSilView, AlarmGorulduView,
    MarketFiyatlarView, LimitDurumuView, GuncelFiyatView,
    GostergeSerileriView, TaramaView, TaramaGostergeView,
    AbonelikDurumuView, AbonelikIptalView, AbonelikUcretsizeGecView,
)
from .api_views_varlik import VarlikHavuzuView, UsdTryKuruView
from .api_views_web_sso import MobilWebGirisTokenView

urlpatterns = [
    # ── Auth ──
    path('auth/register/', KayitView.as_view(), name='api_kayit'),
    path('auth/login/', GirisView.as_view(), name='api_giris'),
    path('auth/refresh/', TokenRefreshView.as_view(), name='api_token_yenile'),
    path('auth/logout/', CikisView.as_view(), name='api_cikis'),
    path('auth/me/', BenKimimView.as_view(), name='api_ben_kimim'),

    # ── Watchlist ──
    path('watchlist/', WatchlistView.as_view(), name='api_watchlist'),
    path('watchlist/<int:oge_id>/', WatchlistSilView.as_view(), name='api_watchlist_sil'),

    # ── Fiyat Alarmları ──
    path('alarmlar/', AlarmlarView.as_view(), name='api_alarmlar'),
    path('alarmlar/<int:alarm_id>/', AlarmSilView.as_view(), name='api_alarmlar_sil'),
    path('alarmlar/gorundu/', AlarmGorulduView.as_view(), name='api_alarmlar_gorundu'),

    # ── Market Fiyatları + Limit ──
    path('market/<str:pazar>/fiyatlar/', MarketFiyatlarView.as_view(), name='api_market_fiyatlar'),
    path('limit-durumu/', LimitDurumuView.as_view(), name='api_limit_durumu'),

    # ── Varlık Havuzu ──
    path('varlik-havuzu/', VarlikHavuzuView.as_view(), name='api_varlik_havuzu'),

    # ── Tek Sembol Güncel Fiyat ──
    path('fiyat/', GuncelFiyatView.as_view(), name='api_guncel_fiyat'),

    # ── Tek Varlık Gösterge Serileri / Analiz Kartları ──
    path('gosterge-serileri/', GostergeSerileriView.as_view(), name='api_gosterge_serileri'),

    # ── Tarama / Screener ──
    path('tarama/<str:pazar>/', TaramaView.as_view(), name='api_tarama'),
    path('tarama-gosterge/<str:pazar>/', TaramaGostergeView.as_view(), name='api_tarama_gosterge'),

    # ── USD/TRY Kuru ──
    path('usd-try-kuru/', UsdTryKuruView.as_view(), name='api_usd_try_kuru'),

    # ── Abonelik (Adım 19 — 🆕) ──
    path('abonelik-durumu/', AbonelikDurumuView.as_view(), name='api_abonelik_durumu'),
    path('abonelik-iptal/', AbonelikIptalView.as_view(), name='api_abonelik_iptal'),
    path('abonelik-ucretsize-gec/', AbonelikUcretsizeGecView.as_view(), name='api_abonelik_ucretsize_gec'),

    # ── Web'e Tek Kullanımlık Otomatik Giriş (Adım 20 — 🆕) ──
    path('mobil-web-giris-token/', MobilWebGirisTokenView.as_view(), name='api_mobil_web_giris_token'),
]
