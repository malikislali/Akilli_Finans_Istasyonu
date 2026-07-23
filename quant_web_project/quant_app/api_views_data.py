"""
quant_app/api_views_data.py — GÜNCELLENDİ (Adım 19)
Bu dosyanın TAMAMINI mevcut api_views_data.py'nin yerine koy.

DEĞİŞİKLİK: AbonelikDurumuView, AbonelikIptalView, AbonelikUcretsizeGecView
eklendi — web'deki abonelik_durumu_api/abonelik_iptal_et/
abonelik_ucretsize_gec ile AYNI mantık, JWT ile.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed

from rest_framework import status, permissions
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import WatchlistItem, FiyatAlarmi, SiteAyari, GunlukAnaliz, Abonelik, AbonelikPlan
from .api_serializers_data import (
    WatchlistOgeSerializer, WatchlistEkleSerializer,
    AlarmSerializer, AlarmEkleSerializer,
)
from . import quant_ml_core as core
from .views import (
    _kullanici_premium_mi,
    _pazar_cache_ile_fiyat_bul,
    market_fiyatlarini_cacheli_getir,
    siralamayi_uygula,
    _gunluk_kayit_getir,
)


def _pazar_cache_ile_veri_bul(sembol, pazar):
    try:
        liste = market_fiyatlarini_cacheli_getir(pazar, '1d')
        for v in liste:
            if v.get('sembol') == sembol and v.get('basarili'):
                return {'fiyat': v.get('fiyat'), 'degisim_yuzde': v.get('degisim_yuzde')}
    except Exception:
        pass
    return {'fiyat': None, 'degisim_yuzde': None}


# =====================================================================
# 💳 ABONELİK (Adım 19 — 🆕)
# =====================================================================
class AbonelikDurumuView(APIView):
    """GET /api/v1/abonelik-durumu/ — web'deki abonelik_durumu_api ile AYNI."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        try:
            ab = request.user.abonelik
            return Response({
                'basarili': True,
                'plan': ab.plan.ad,
                'plan_gorunen': ab.plan.get_ad_display(),
                'durum': ab.durum,
                'durum_gorunen': ab.get_durum_display(),
                'bitis': ab.bitis.strftime('%d.%m.%Y') if ab.bitis else None,
                'gecerli_mi': ab.gecerli_mi,
                'premium_mi': ab.premium_mi,
            })
        except Abonelik.DoesNotExist:
            return Response({
                'basarili': True, 'plan': 'ucretsiz', 'plan_gorunen': 'Ücretsiz',
                'durum': None, 'durum_gorunen': None, 'bitis': None,
                'gecerli_mi': False, 'premium_mi': False,
            })


class AbonelikIptalView(APIView):
    """POST /api/v1/abonelik-iptal/ — web'deki abonelik_iptal_et ile AYNI."""
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        try:
            ab = request.user.abonelik
            ab.durum = 'iptal'
            ab.save()
            return Response({'basarili': True})
        except Abonelik.DoesNotExist:
            return Response({'basarili': False, 'hata_mesaji': 'Aktif bir aboneliğiniz yok.'},
                             status=status.HTTP_400_BAD_REQUEST)


class AbonelikUcretsizeGecView(APIView):
    """POST /api/v1/abonelik-ucretsize-gec/ — web'deki abonelik_ucretsize_gec ile AYNI."""
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        ucretsiz_plan = AbonelikPlan.objects.get(ad='ucretsiz')
        ab, _ = Abonelik.objects.get_or_create(user=request.user, defaults={'plan': ucretsiz_plan})
        ab.plan = ucretsiz_plan
        ab.durum = 'aktif'
        ab.bitis = None
        ab.save()
        return Response({'basarili': True})


# =====================================================================
# 🔎 TARAMA — Hafif Filtreler
# =====================================================================
class TaramaView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    FILTRE_FONKSIYONLARI = {
        'degisim_pozitif':     lambda s: s['degisim_yuzde'] > 0,
        'degisim_negatif':     lambda s: s['degisim_yuzde'] < 0,
        'degisim_guclu':       lambda s: s['degisim_yuzde'] > 3,
        'dusus_guclu':         lambda s: s['degisim_yuzde'] < -3,
        'hacim_artan':         lambda s: s['hacim_degisim_yuzde'] > 0,
        'hacim_azalan':        lambda s: s['hacim_degisim_yuzde'] < 0,
        'fiyat_dusuk_degisim': lambda s: abs(s['degisim_yuzde']) < 1,
    }

    def get(self, request, pazar):
        if pazar not in core.VARLIK_HAVUZU:
            return Response({'basarili': False, 'hata_mesaji': f"Geçersiz pazar: {pazar}"},
                             status=status.HTTP_400_BAD_REQUEST)

        ayar = SiteAyari.get()
        if not _kullanici_premium_mi(request):
            kayit = _gunluk_kayit_getir(request)
            if kayit.tarama_adet >= ayar.ucretsiz_gunluk_tarama_limiti:
                return Response({
                    'basarili': False, 'limit_asimi': True, 'limit_turu': 'tarama',
                    'hata_mesaji': "Günlük deneyim limiti aşıldı. Sınırsız bir deneyim için Ücretli Planlarımıza geçebilirsiniz.",
                }, status=status.HTTP_429_TOO_MANY_REQUESTS)
            GunlukAnaliz.objects.filter(pk=kayit.pk).update(tarama_adet=kayit.tarama_adet + 1)

        interval = request.query_params.get('interval', '1d').strip()
        filtreler_str = request.query_params.get('filtreler', '').strip()
        filtreler = [f.strip() for f in filtreler_str.split(',') if f.strip()]

        sonuclar = market_fiyatlarini_cacheli_getir(pazar, interval)
        basarili = [s for s in sonuclar if s['basarili']]

        if not basarili:
            return Response({
                'basarili': True, 'pazar': pazar, 'sonuc_sayisi': 0, 'toplam_varlik': 0,
                'varliklar': [], 'mesaj': 'Veri henüz yüklenmedi, Anasayfa sekmesini açıp veri yüklenmesini bekleyin.',
            })

        hacim_listesi = [s['hacim'] for s in basarili if s['hacim'] > 0]
        hacim_ortalama = sum(hacim_listesi) / len(hacim_listesi) if hacim_listesi else 0

        sonuc = basarili
        gecersiz_filtreler = []
        for filtre in filtreler:
            if filtre == 'hacim_yuksek':
                sonuc = [s for s in sonuc if s['hacim'] > hacim_ortalama]
            elif filtre in self.FILTRE_FONKSIYONLARI:
                sonuc = [s for s in sonuc if self.FILTRE_FONKSIYONLARI[filtre](s)]
            elif filtre:
                gecersiz_filtreler.append(filtre)

        sonuc.sort(key=lambda x: abs(x['degisim_yuzde']), reverse=True)

        return Response({
            'basarili': True, 'pazar': pazar, 'interval': interval,
            'uygulanan_filtreler': filtreler, 'gecersiz_filtreler': gecersiz_filtreler,
            'sonuc_sayisi': len(sonuc), 'toplam_varlik': len(basarili),
            'varliklar': sonuc,
        })


# =====================================================================
# 🔎 TARAMA — Gösterge Tabanlı Filtreler (Trend/Momentum)
# =====================================================================
class TaramaGostergeView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, pazar):
        if pazar not in core.VARLIK_HAVUZU:
            return Response({'basarili': False, 'hata_mesaji': f"Geçersiz pazar: {pazar}"},
                             status=status.HTTP_400_BAD_REQUEST)

        ayar = SiteAyari.get()
        if not _kullanici_premium_mi(request):
            kayit = _gunluk_kayit_getir(request)
            if kayit.tarama_adet >= ayar.ucretsiz_gunluk_tarama_limiti:
                return Response({
                    'basarili': False, 'limit_asimi': True, 'limit_turu': 'tarama',
                    'hata_mesaji': "Günlük deneyim limiti aşıldı. Sınırsız bir deneyim için Ücretli Planlarımıza geçebilirsiniz.",
                }, status=status.HTTP_429_TOO_MANY_REQUESTS)
            GunlukAnaliz.objects.filter(pk=kayit.pk).update(tarama_adet=kayit.tarama_adet + 1)

        interval = request.query_params.get('interval', '1d').strip()
        filtreler_str = request.query_params.get('filtreler', '').strip()
        filtreler = [f.strip() for f in filtreler_str.split(',') if f.strip()]
        try:
            rsi_deger = float(request.query_params.get('rsi_deger', '50'))
        except ValueError:
            rsi_deger = 50.0

        semboller = core.VARLIK_HAVUZU[pazar]

        def _gosterge_filtrele(g):
            n = len(g.close)
            if n < 3:
                return False
            for filtre in filtreler:
                if filtre == 'macd_yukari':
                    if not (len(g.macd) >= 2 and len(g.macd_sinyal) >= 2 and
                            g.macd[-2] < g.macd_sinyal[-2] and g.macd[-1] > g.macd_sinyal[-1]):
                        return False
                elif filtre == 'macd_sifir_ustune':
                    if not (len(g.macd) >= 2 and g.macd[-2] < 0 and g.macd[-1] > 0):
                        return False
                elif filtre == 'ema_kesisim':
                    if not (len(g.ema_20) >= 2 and len(g.ema_50) >= 2 and
                            g.ema_20[-2] < g.ema_50[-2] and g.ema_20[-1] > g.ema_50[-1]):
                        return False
                elif filtre == 'bollinger_alt_kirma':
                    if not (len(g.close) >= 2 and len(g.bollinger_alt) >= 2 and
                            g.close[-2] < g.bollinger_alt[-2] and g.close[-1] > g.bollinger_alt[-1]):
                        return False
                elif filtre == 'rsi_asiri_satim':
                    if not (len(g.rsi) >= 2 and g.rsi[-2] < 30 and g.rsi[-1] > 30):
                        return False
                elif filtre == 'rsi_asiri_alim':
                    if not (len(g.rsi) >= 2 and g.rsi[-2] > 70 and g.rsi[-1] < 70):
                        return False
                elif filtre == 'rsi_deger_alti':
                    if not (len(g.rsi) >= 1 and g.rsi[-1] < rsi_deger):
                        return False
                elif filtre == 'rsi_deger_ustu':
                    if not (len(g.rsi) >= 1 and g.rsi[-1] > rsi_deger):
                        return False
            return True

        def _tek_varlik(sembol):
            g = core.gosterge_serileri_getir(sembol, pazar, interval)
            if g.basarili and _gosterge_filtrele(g):
                return {
                    'sembol': sembol, 'basarili': True,
                    'fiyat': float(g.close[-1]) if g.close else 0,
                    'degisim_yuzde': float(((g.close[-1] - g.close[-2]) / g.close[-2] * 100)
                                            if len(g.close) >= 2 and g.close[-2] != 0 else 0),
                    'hacim': float(g.volume[-1]) if g.volume else 0,
                    'rsi': float(g.rsi[-1]) if g.rsi else 0,
                    'macd': float(g.macd[-1]) if g.macd else 0,
                }
            return None

        sonuclar = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(_tek_varlik, s): s for s in semboller}
            for future in as_completed(futures):
                try:
                    r = future.result()
                    if r:
                        sonuclar.append(r)
                except Exception:
                    continue

        sonuclar.sort(key=lambda x: abs(x['degisim_yuzde']), reverse=True)

        return Response({
            'basarili': True, 'pazar': pazar, 'interval': interval,
            'sonuc_sayisi': len(sonuclar), 'toplam_varlik': len(semboller),
            'varliklar': sonuclar,
        })


# =====================================================================
# 💲 TEK SEMBOL GÜNCEL FİYAT
# =====================================================================
class GuncelFiyatView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        pazar = request.query_params.get('pazar', '').strip()
        sembol = request.query_params.get('sembol', '').strip()

        if not pazar or not sembol:
            return Response({'basarili': False, 'hata_mesaji': 'pazar ve sembol parametreleri zorunlu.'},
                             status=status.HTTP_400_BAD_REQUEST)
        if pazar not in core.VARLIK_HAVUZU:
            return Response({'basarili': False, 'hata_mesaji': f"Geçersiz pazar: {pazar}"},
                             status=status.HTTP_400_BAD_REQUEST)

        fiyat = _pazar_cache_ile_fiyat_bul(sembol, pazar)
        return Response({'basarili': True, 'sembol': sembol, 'pazar': pazar, 'fiyat': fiyat})


# =====================================================================
# 📈 TEK VARLIK GÖSTERGE SERİLERİ
# =====================================================================
class GostergeSerileriView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        ayar = SiteAyari.get()
        premium = _kullanici_premium_mi(request)

        if not premium:
            kayit = _gunluk_kayit_getir(request)
            if kayit.adet >= ayar.ucretsiz_gunluk_analiz_limiti:
                return Response({
                    'basarili': False, 'limit_asimi': True, 'limit_turu': 'analiz',
                    'hata_mesaji': "Günlük deneyim limiti aşıldı. Sınırsız bir deneyim için Ücretli Planlarımıza geçebilirsiniz.",
                }, status=status.HTTP_429_TOO_MANY_REQUESTS)
            GunlukAnaliz.objects.filter(pk=kayit.pk).update(adet=kayit.adet + 1)

        sembol = request.query_params.get('sembol', '').strip()
        pazar = request.query_params.get('pazar', '').strip()
        interval = request.query_params.get('interval', '1d').strip()

        if not sembol or not pazar:
            return Response({'basarili': False, 'hata_mesaji': 'sembol/pazar parametreleri eksik.'},
                             status=status.HTTP_400_BAD_REQUEST)
        if pazar not in core.VARLIK_HAVUZU:
            return Response({'basarili': False, 'hata_mesaji': f"Geçersiz pazar: {pazar}"},
                             status=status.HTTP_400_BAD_REQUEST)

        sonuc = core.gosterge_serileri_getir(sembol, pazar, interval)
        if not sonuc.basarili:
            return Response({'basarili': False, 'hata_mesaji': sonuc.hata_mesaji})

        return Response({
            'basarili': True, 'sembol': sonuc.sembol, 'pazar': sonuc.pazar,
            'tarihler': sonuc.tarihler,
            'open': sonuc.open, 'high': sonuc.high, 'low': sonuc.low, 'close': sonuc.close, 'volume': sonuc.volume,
            'ema_20': sonuc.ema_20, 'ema_50': sonuc.ema_50, 'ema_100': sonuc.ema_100, 'sma_200': sonuc.sma_200,
            'bollinger_ust': sonuc.bollinger_ust, 'bollinger_orta': sonuc.bollinger_orta, 'bollinger_alt': sonuc.bollinger_alt,
            'rsi': sonuc.rsi, 'macd': sonuc.macd, 'macd_sinyal': sonuc.macd_sinyal,
            'stoch_k': sonuc.stoch_k, 'stoch_d': sonuc.stoch_d,
            'wt1': sonuc.wt1, 'wt2': sonuc.wt2, 'cci': sonuc.cci, 'atr': sonuc.atr,
            'ath': sonuc.ath,
        })


# =====================================================================
# 📌 WATCHLIST
# =====================================================================
class WatchlistView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        ogeler = list(WatchlistItem.objects.filter(user=request.user))
        for o in ogeler:
            veri = _pazar_cache_ile_veri_bul(o.sembol, o.pazar)
            o._guncel_fiyat = veri['fiyat']
            o._degisim_yuzde = veri['degisim_yuzde']
        return Response({'basarili': True, 'ogeler': WatchlistOgeSerializer(ogeler, many=True).data})

    def post(self, request):
        giris = WatchlistEkleSerializer(data=request.data)
        giris.is_valid(raise_exception=True)
        sembol = giris.validated_data['sembol'].strip()
        pazar = giris.validated_data['pazar'].strip()

        if pazar not in core.VARLIK_HAVUZU:
            return Response({'basarili': False, 'hata_mesaji': f"Geçersiz pazar: {pazar}"},
                             status=status.HTTP_400_BAD_REQUEST)

        zaten_var_mi = WatchlistItem.objects.filter(user=request.user, sembol=sembol, pazar=pazar).exists()
        if not zaten_var_mi and not _kullanici_premium_mi(request):
            ayar = SiteAyari.get()
            mevcut_sayi = WatchlistItem.objects.filter(user=request.user).count()
            if mevcut_sayi >= ayar.ucretsiz_takip_listesi_limiti:
                return Response({
                    'basarili': False, 'limit_asimi': True, 'limit_turu': 'takip_listesi',
                    'hata_mesaji': f"{ayar.ucretsiz_takip_listesi_limiti} varlık limitine sahipsiniz. Sınırsız varlık takibi için Ücretli Planlarımıza geçebilirsiniz.",
                }, status=status.HTTP_429_TOO_MANY_REQUESTS)

        oge, olusturuldu_mu = WatchlistItem.objects.get_or_create(
            user=request.user, sembol=sembol, pazar=pazar,
            defaults={'siralama': WatchlistItem.objects.filter(user=request.user).count()},
        )
        veri = _pazar_cache_ile_veri_bul(oge.sembol, oge.pazar)
        oge._guncel_fiyat = veri['fiyat']
        oge._degisim_yuzde = veri['degisim_yuzde']
        return Response({
            'basarili': True, 'olusturuldu_mu': olusturuldu_mu,
            'oge': WatchlistOgeSerializer(oge).data,
        }, status=status.HTTP_201_CREATED if olusturuldu_mu else status.HTTP_200_OK)


class WatchlistSilView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def delete(self, request, oge_id):
        try:
            oge = WatchlistItem.objects.get(id=oge_id, user=request.user)
        except WatchlistItem.DoesNotExist:
            return Response({'basarili': False, 'hata_mesaji': 'Öğe bulunamadı.'},
                             status=status.HTTP_404_NOT_FOUND)
        oge.delete()
        return Response({'basarili': True})


# =====================================================================
# 🔔 FİYAT ALARMLARI
# =====================================================================
class AlarmlarView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        alarmlar = list(FiyatAlarmi.objects.filter(user=request.user))
        for a in alarmlar:
            a._guncel_fiyat = _pazar_cache_ile_fiyat_bul(a.sembol, a.pazar)
        return Response({'basarili': True, 'alarmlar': AlarmSerializer(alarmlar, many=True).data})

    def post(self, request):
        giris = AlarmEkleSerializer(data=request.data)
        giris.is_valid(raise_exception=True)
        v = giris.validated_data

        if v['pazar'] not in core.VARLIK_HAVUZU:
            return Response({'basarili': False, 'hata_mesaji': 'Geçersiz pazar.'},
                             status=status.HTTP_400_BAD_REQUEST)

        if not _kullanici_premium_mi(request):
            ayar = SiteAyari.get()
            aktif_sayisi = FiyatAlarmi.objects.filter(user=request.user, aktif=True, tetiklendi_mi=False).count()
            if aktif_sayisi >= ayar.ucretsiz_alarm_limiti:
                return Response({
                    'basarili': False, 'limit_asimi': True, 'limit_turu': 'alarm',
                    'hata_mesaji': f"{ayar.ucretsiz_alarm_limiti} alarm limitine sahipsiniz. Sınırsız alarm için Ücretli Planlarımıza geçebilirsiniz.",
                }, status=status.HTTP_429_TOO_MANY_REQUESTS)

        guncel_fiyat = _pazar_cache_ile_fiyat_bul(v['sembol'], v['pazar'])
        alarm = FiyatAlarmi.objects.create(
            user=request.user, sembol=v['sembol'], pazar=v['pazar'],
            hedef_fiyat=v['hedef_fiyat'], yon=v['yon'], olusturulma_fiyati=guncel_fiyat,
        )
        alarm._guncel_fiyat = guncel_fiyat
        return Response({'basarili': True, 'alarm': AlarmSerializer(alarm).data}, status=status.HTTP_201_CREATED)


class AlarmSilView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def delete(self, request, alarm_id):
        FiyatAlarmi.objects.filter(id=alarm_id, user=request.user).delete()
        return Response({'basarili': True})


class AlarmGorulduView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        FiyatAlarmi.objects.filter(user=request.user, tetiklendi_mi=True, goruldu_mu=False).update(goruldu_mu=True)
        return Response({'basarili': True})


# =====================================================================
# 📊 MARKET FİYATLARI
# =====================================================================
class MarketFiyatlarView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, pazar):
        if pazar not in core.VARLIK_HAVUZU:
            return Response({'basarili': False, 'hata_mesaji': f"Geçersiz pazar: {pazar}"},
                             status=status.HTTP_400_BAD_REQUEST)

        interval = request.query_params.get('interval', '1d').strip()
        siralama = request.query_params.get('siralama', '').strip()

        sonuclar = market_fiyatlarini_cacheli_getir(pazar, interval)
        sonuclar = siralamayi_uygula(sonuclar, siralama)

        return Response({'basarili': True, 'pazar': pazar, 'interval': interval, 'varliklar': sonuclar})


# =====================================================================
# 🚦 LİMİT DURUMU
# =====================================================================
class LimitDurumuView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        premium = _kullanici_premium_mi(request)
        if premium:
            return Response({'basarili': True, 'premium': True})

        ayar = SiteAyari.get()
        kayit = _gunluk_kayit_getir(request)
        takip_sayisi = WatchlistItem.objects.filter(user=request.user).count()
        alarm_aktif_sayisi = FiyatAlarmi.objects.filter(user=request.user, aktif=True, tetiklendi_mi=False).count()

        return Response({
            'basarili': True,
            'premium': False,
            'analiz': {
                'kullanilan': kayit.adet, 'limit': ayar.ucretsiz_gunluk_analiz_limiti,
                'kalan': max(0, ayar.ucretsiz_gunluk_analiz_limiti - kayit.adet),
            },
            'tarama': {
                'kullanilan': kayit.tarama_adet, 'limit': ayar.ucretsiz_gunluk_tarama_limiti,
                'kalan': max(0, ayar.ucretsiz_gunluk_tarama_limiti - kayit.tarama_adet),
            },
            'lab_analiz': {
                'kullanilan': kayit.lab_analiz_adet, 'limit': ayar.ucretsiz_gunluk_lab_analiz_limiti,
                'kalan': max(0, ayar.ucretsiz_gunluk_lab_analiz_limiti - kayit.lab_analiz_adet),
            },
            'takip_listesi': {
                'kullanilan': takip_sayisi, 'limit': ayar.ucretsiz_takip_listesi_limiti,
                'kalan': max(0, ayar.ucretsiz_takip_listesi_limiti - takip_sayisi),
            },
            'alarm': {
                'kullanilan': alarm_aktif_sayisi, 'limit': ayar.ucretsiz_alarm_limiti,
                'kalan': max(0, ayar.ucretsiz_alarm_limiti - alarm_aktif_sayisi),
            },
        })
