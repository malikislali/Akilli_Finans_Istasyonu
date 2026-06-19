from django.http import JsonResponse
from django.shortcuts import render
from django.utils import timezone
from .models import QuantSignalCache, Portfolio
from .quant_ml_core import calistir_quant_analiz
import json
from datetime import timedelta

# =====================================================================
# 📊 1. GÖRÜNÜM (VIEW): SAYFA AÇILDIĞINDA OTOMATİK ÇALIŞAN KOKPİT
# =====================================================================
def dashboard_kokpit(request):
    """
    Kullanıcı web sitesini açtığı an (hiçbir butona basmadan) tetiklenen,
    ML Ensemble tahminlerini ve Plotly grafik verilerini önbellek korumalı
    olarak frontend şablonuna basan ana üs hoca.
    """
    # Varsayılan olarak BTC-USD paritesini analiz edelim
    sembol = request.GET.get('sembol', 'BTC-USD')
    pazar = request.GET.get('pazar', 'KRIPTO')
    period = request.GET.get('period', '1mo')
    interval = request.GET.get('interval', '15m')
    
    # 🛡️ 15 Dakikalık Önbellek (Cache) Güvenlik Duvarı
    zaman_esigi = timezone.now() - timedelta(minutes=15)
    cache_kaydi = QuantSignalCache.objects.filter(sembol=sembol, son_guncellenme__gte=zaman_esigi).first()
    
    if cache_kaydi:
        # Eğer taze veri varsa yfinance'i yormuyoruz, direkt DB'den çekiyoruz hoca!
        analiz_sonucu = {
            "durum": "basarili",
            "meta": {"sembol": cache_kaydi.sembol, "pazar": cache_kaydi.pazar, "period": cache_kaydi.period, "interval": interval},
            "anlik_veri": {
                "fiyat": cache_kaydi.son_fiyat,
                "degisim_24s": cache_kaydi.degisim_24s,
                "atr": cache_kaydi.atr_gucu,
            },
            "tahmin_raporu": {
                "karar": cache_kaydi.konsensüs_karari,
                "boga_ihtimali": cache_kaydi.boga_ihtimali,
                "ayi_ihtimali": cache_kaydi.ayi_ihtimali,
            },
            "performans_metrikleri": {
                "ensemble_accuracy": cache_kaydi.ensemble_accuracy,
            },
            "grafik_verisi": json.loads(cache_kaydi.grafik_verisi_json)
        }
        veri_kaynagi = "Veritabanı Önbelleği (Cache)"
    else:
        # Önbellek yoksa veya eskidiyse ML motorunu can yakıcı güçle ateşle!
        analiz_sonucu = calistir_quant_analiz(sembol, pazar, period, interval)
        veri_kaynagi = "Canlı Yapay Zeka Konsorsiyumu (ML Core Engine)"
        
        # Gelen taze sonucu veritabanına mühürle (Bir sonraki kullanıcı saniyede okusun)
        if analiz_sonucu.get("durum") == "basarili":
            QuantSignalCache.objects.update_or_create(
                sembol=sembol,
                defaults={
                    'pazar': pazar,
                    'period': period,
                    'son_fiyat': analiz_sonucu['anlik_veri']['fiyat'],
                    'degisim_24s': analiz_sonucu['anlik_veri']['degisim_24s'],
                    'atr_gucu': analiz_sonucu['anlik_veri']['atr'],
                    'konsensüs_karari': analiz_sonucu['tahmin_raporu']['karar'],
                    'boga_ihtimali': analiz_sonucu['tahmin_raporu']['boga_ihtimali'],
                    'ayi_ihtimali': analiz_sonucu['tahmin_raporu']['ayi_ihtimali'],
                    'ensemble_accuracy': analiz_sonucu['performans_metrikleri']['ensemble_accuracy'],
                    'grafik_verisi_json': json.dumps(analiz_sonucu['grafik_verisi']),
                }
            )
            
    # Kullanıcının cüzdan portföy verilerini de yanına katık edelim
    kullanici_portfoyu = []
    if request.user.is_authenticated:
        kullanici_portfoyu = Portfolio.objects.filter(user=request.user)

    context = {
        "analiz": analiz_sonucu,
        "portfoy": kullanici_portfoyu,
        "kaynak": veri_kaynagi
    }
    
    # Verileri HTML şablonumuza üflüyoruz hoca
    return render(request, 'quant_app/dashboard.html', context)


# =====================================================================
# ⚡ 2. GÖRÜNÜM (VIEW): DIŞ DÜNYA VE ARDUINO İÇİN JET JSON API
# =====================================================================
def sinyal_api_endpoint(request, sembol):
    """
    İleride yapacağımız o siber masa üstü hatırlatıcısının (ESP32/Arduino)
    veya n8n botunun doğrudan bağlanıp saf JSON çekeceği gizli kapı hoca!
    """
    cache_kaydi = QuantSignalCache.objects.filter(sembol=sembol).first()
    if cache_kaydi:
        data = {
            "sembol": cache_kaydi.sembol,
            "fiyat": cache_kaydi.son_fiyat,
            "karar": cache_kaydi.konsensüs_karari,
            "boga_skoru": f"%{cache_kaydi.boga_ihtimali:.1f}",
            "son_guncelleme": cache_kaydi.son_guncellenme.strftime('%H:%M:%S')
        }
        return JsonResponse(data, safe=False)
    return JsonResponse({"hata": "Bu sembole ait sinyal kaydi bulunamadi."}, status=404)