"""
quant_app/api_views_varlik.py — GÜNCELLENDİ (Adım 16)
Bu dosyanın TAMAMINI mevcut api_views_varlik.py'nin yerine koy.

DEĞİŞİKLİK: UsdTryKuruView eklendi — web'deki usd_try_kuru_api ile AYNI
kaynağı (core.usd_try_kuru_getir, 10 dakikalık Django cache) kullanır.
Mobil app'teki Karma/$/₺ para birimi dönüşümü bunu kullanacak.
"""
from rest_framework import permissions
from rest_framework.response import Response
from rest_framework.views import APIView

from . import quant_ml_core as core

PAZAR_GORUNUM_BILGISI = {
    "KRIPTO": {"emoji": "🟠", "ad": "Kripto", "renk": "#F7931A"},
    "TR_HISSE": {"emoji": "🔵", "ad": "TR Hisse", "renk": "#3B82F6"},
    "ABD_HISSE": {"emoji": "🟢", "ad": "ABD Hisse", "renk": "#22C55E"},
    "EMTIA": {"emoji": "🟡", "ad": "Emtia", "renk": "#EAB308"},
}


class VarlikHavuzuView(APIView):
    """GET /api/v1/varlik-havuzu/"""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        return Response({
            'basarili': True,
            'varlik_havuzu': core.VARLIK_HAVUZU,
            'pazar_bilgisi': PAZAR_GORUNUM_BILGISI,
            'interval_secenekleri': core.DISPLAY_INTERVALS_BY_MARKET,
        })


class UsdTryKuruView(APIView):
    """GET /api/v1/usd-try-kuru/
    Web'deki usd_try_kuru_api ile AYNI — 10 dakikalık cache'ten okur,
    yoksa Yahoo'dan çeker. Karma/$/₺ dönüşümü için kullanılır."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        kur = core.usd_try_kuru_getir()
        return Response({'basarili': True, 'kur': kur})
