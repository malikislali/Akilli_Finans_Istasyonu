"""
quant_app/api_views_web_sso.py — 🆕 Adım 20

Mobil app'te zaten JWT ile giriş yapmış bir kullanıcının, "Web
Sitesinden Plan Yönet" gibi bir linke dokununca tarayıcıda TEKRAR
kullanıcı adı/şifre girmesine gerek kalmadan web tarafında da otomatik
oturum açılmasını sağlar.

NASIL ÇALIŞIYOR:
1) Mobil app bu endpoint'e (JWT ile) istek atar → 60 saniye geçerli,
   TEK KULLANIMLIK rastgele bir token üretilir, Django cache'ine
   (kullanıcı id'siyle eşlenmiş) yazılır.
2) Mobil app, dönen token'ı `https://piyasapusulam.com/mobil-giris/<token>/?next=/plan-sec/`
   şeklinde bir URL'e ekleyip telefonun TARAYICISINDA açar (WebView değil
   — Play Store/App Store'un "consumption-only app" kuralına uygun
   kalmak için önemli, ödeme akışı hep gerçek tarayıcıda kalıyor).
3) `quant_app/views_web_sso.py`'deki düz Django view, bu token'ı
   cache'ten okur, doğruysa Django SESSION login yapar (web'in normal
   oturum açma mekanizmasıyla AYNI), token'ı hemen siler (tek
   kullanımlık), sonra `next` parametresindeki sayfaya yönlendirir.

GÜVENLİK: Token 60 saniye içinde kullanılmazsa kendiliğinden geçersiz
olur (cache timeout). Kullanılınca hemen silinir — aynı token ikinci
kez işe yaramaz.
"""
import secrets

from django.core.cache import cache
from rest_framework import permissions
from rest_framework.response import Response
from rest_framework.views import APIView

MOBIL_GIRIS_TOKEN_SURESI_SANIYE = 60
MOBIL_GIRIS_CACHE_ANAHTAR_ONEKI = 'mobil_web_giris_'


class MobilWebGirisTokenView(APIView):
    """POST /api/v1/mobil-web-giris-token/"""
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        token = secrets.token_urlsafe(32)
        cache.set(f'{MOBIL_GIRIS_CACHE_ANAHTAR_ONEKI}{token}', request.user.id,
                   timeout=MOBIL_GIRIS_TOKEN_SURESI_SANIYE)
        return Response({'basarili': True, 'token': token})
