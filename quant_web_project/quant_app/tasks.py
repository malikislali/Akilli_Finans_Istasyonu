"""
tasks.py — Celery arka plan görevleri.

Şu an tek görev: popüler pazarların fiyat listesi cache'ini, süresi
dolmadan önce arka planda yeniden doldurmak (market_fiyatlarini_cacheli_getir
zaten var olan cache mantığını AYNEN kullanır — sadece TETİKLEYEN artık
bir kullanıcı isteği değil, Celery-beat).
"""

from celery import shared_task
from .views import market_fiyatlarini_cacheli_getir
from . import quant_ml_core as core


# Her pazar için ön-ısıtılacak periyot(lar). Şimdilik sadece '1d'
# (varsayılan/en çok kullanılan) — istersen sonra '4h' gibi ek
# periyotlar da buraya eklenebilir.
ON_ISITILACAK_PERIYOTLAR = ['15m', '30m', '60m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '1wk']
@shared_task(name='quant_app.market_cache_yenile')
def market_cache_yenile():
    """
    Her pazar × her periyot kombinasyonu için market_fiyatlarini_cacheli_getir'i
    çağırır. Bu fonksiyon zaten kendi içinde cache okuma/yazma yapıyor —
    burada tek yaptığımız, bunu KULLANICI BEKLEMEDEN, periyodik olarak
    tetiklemek.
    """
   
   # Herhangi bir sembol için Yahoo 429/rate-limit hatası verirse, anında 15 dakikalık soğuma moduna geçilir
   # Soğuma süresince, hem canlı kullanıcı istekleri hem de Celery ön-ısıtma turları Yahoo'ya hiç istek atmaz — mevcut cache'i kullanmaya devam eder (veri biraz eskir ama site çökmez)
#15 dakika sonra otomatik olarak normale döner, tekrar deneme başlar
#Tek bir sembolün rate-limit'e takılması, tüm sistemi (diğer pazarlar dahil) korumaya alır — çünkü rate limit genelde IP bazlı uygulanıyor, tek sembole özel değil
    if core._yahoo_rate_limitli_mi():
        return {'atlandi': 'Yahoo rate-limit soğuma süresinde, bu tur atlandı.'}

    sonuclar = {}
    for pazar in core.VARLIK_HAVUZU.keys():

        for interval in ON_ISITILACAK_PERIYOTLAR:
            try:
                veri = market_fiyatlarini_cacheli_getir(pazar, interval)
                sonuclar[f"{pazar}_{interval}"] = len(veri)
            except Exception as exc:
                sonuclar[f"{pazar}_{interval}"] = f"HATA: {exc}"
    return sonuclar

@shared_task(name='quant_app.analiz_hesapla_task', bind=True)
def analiz_hesapla_task(self, sembol, pazar, interval):
    """
    quant_ml_core.analiz_yap()'ı arka planda çalıştırır. Mevcut
    views.py'deki yardımcı fonksiyonları (JSON'a çevirme, cache'e yazma)
    AYNEN kullanır — mantık tekrarlanmıyor, sadece TETİKLEYEN değişiyor.
    """
    from . import quant_ml_core as core
    from .views import _analiz_sonucunu_json_a_cevir, _analiz_sonucunu_cache_e_yaz

    sonuc = core.analiz_yap(sembol, pazar, interval)

    if sonuc.basarili and not sonuc.veri_yetersiz:
        _analiz_sonucunu_cache_e_yaz(sonuc)

    return _analiz_sonucunu_json_a_cevir(sonuc)

@shared_task(name='quant_app.usd_try_kuru_yenile')
def usd_try_kuru_yenile():
    kur = core.usd_try_kuru_getir()
    return {'kur': kur}


@shared_task(name='quant_app.alarmlari_kontrol_et')
def alarmlari_kontrol_et():
    """Aktif, henüz tetiklenmemiş tüm alarmları güncel fiyatla karşılaştırır.
    Hedefe ulaşılınca: tetiklendi_mi=True, aktif=False, e-posta gönderir."""
    from .models import FiyatAlarmi
    from . import quant_ml_core as core
    from django.core.mail import send_mail
    from django.conf import settings as dj_settings
    from django.utils import timezone

    aktif_alarmlar = FiyatAlarmi.objects.filter(aktif=True, tetiklendi_mi=False).select_related('user')
    tetiklenen_sayisi = 0

    for alarm in aktif_alarmlar:
        try:
            sonuc = core.hafif_fiyat_getir(alarm.sembol, alarm.pazar, interval="1d")
            if not sonuc.basarili or sonuc.fiyat == 0:
                continue

            guncel_fiyat = sonuc.fiyat
            hedef = float(alarm.hedef_fiyat)
            tetiklendi = (
                (alarm.yon == 'ustune' and guncel_fiyat >= hedef) or
                (alarm.yon == 'altina' and guncel_fiyat <= hedef)
            )

            if tetiklendi:
                alarm.tetiklendi_mi = True
                alarm.aktif = False
                alarm.goruldu_mu = False
                alarm.tetiklenme_tarihi = timezone.now()
                alarm.save()
                tetiklenen_sayisi += 1

                yon_metni = "üstüne çıktı" if alarm.yon == 'ustune' else "altına indi"
                if alarm.user.email:
                    try:
                        send_mail(
                            subject=f"Piyasa Pusulam — {alarm.sembol} Alarmınız Tetiklendi",
                            message=(
                                f"Merhaba {alarm.user.username},\n\n"
                                f"{alarm.sembol} fiyatı, belirlediğiniz {hedef} hedefinin {yon_metni} "
                                f"(güncel fiyat: {guncel_fiyat}).\n\nPiyasa Pusulam Ekibi"
                            ),
                            from_email=dj_settings.DEFAULT_FROM_EMAIL,
                            recipient_list=[alarm.user.email],
                            fail_silently=True,
                        )
                    except Exception:
                        pass
        except Exception:
            continue

    return {'tetiklenen': tetiklenen_sayisi}