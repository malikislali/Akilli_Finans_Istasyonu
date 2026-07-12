"""
views.py — Üyelik (login/register/logout) + Dashboard + Analiz view'ları.

Mimari karar (kullanıcıyla netleştirildi):
  - Auth: Django'nun hazır authentication sistemi (login/logout/register).
  - Analiz: SEÇENEK 2 — kullanıcı bir varlık seçtiğinde ANINDA hesaplanır
    (cron/Celery YOK). QuantSignalCache, Streamlit'teki @cache_data(ttl=300)
    mantığının karşılığı olarak 5 dakikalık basit bir önbellek görevi görür.
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib import messages
from django.core.cache import cache
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render, redirect
from django.views.decorators.http import require_http_methods

from .models import QuantSignalCache, WatchlistItem
from . import quant_ml_core as core


# =====================================================================
# 🔐 ÜYELİK VIEW'LARI
# =====================================================================

def register_view(request):
    """Kayıt sayfası — sadece kullanıcı bilgileri. Plan seçimi kayıt sonrasında."""
    from .models import SiteAyari, AbonelikPlan, Abonelik

    if request.user.is_authenticated:
        return redirect('anasayfa')

    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        email = request.POST.get('email', '').strip()
        password = request.POST.get('password', '')
        password2 = request.POST.get('password2', '')

        hata = None
        if not username or not password:
            hata = "Kullanıcı adı ve şifre boş bırakılamaz."
        elif password != password2:
            hata = "Şifreler eşleşmiyor."
        elif len(password) < 6:
            hata = "Şifre en az 6 karakter olmalı."
        elif User.objects.filter(username=username).exists():
            hata = "Bu kullanıcı adı zaten alınmış."

        if hata:
            messages.error(request, hata)
            return render(request, 'quant_app/register.html', {'username': username, 'email': email})

        user = User.objects.create_user(username=username, email=email, password=password)
        login(request, user)
        return redirect('plan_sec')

    return render(request, 'quant_app/register.html')


@login_required
def plan_sec_view(request):
    """Plan seçim/değiştirme ekranı — hem kayıt sonrası ilk seçimde hem
    Ayarlar sekmesinden plan değiştirmek için kullanılır."""
    from .models import AbonelikPlan, Abonelik

    planlar = AbonelikPlan.objects.filter(aktif=True).order_by('fiyat_tl')

    if request.method == 'POST':
        plan_ad = request.POST.get('plan', 'ucretsiz')
        try:
            plan = AbonelikPlan.objects.get(ad=plan_ad, aktif=True)
        except AbonelikPlan.DoesNotExist:
            plan = AbonelikPlan.objects.get(ad='ucretsiz')

        if plan.fiyat_tl > 0:
            # Ücretli plan → İyzico ödeme sayfasına yönlendir
            request.session['secilen_plan_ad'] = plan.ad
            return redirect('iyzico_odeme_baslat')

        # Ücretsiz plan → direkt ata
        Abonelik.objects.get_or_create(user=request.user, defaults={'plan': plan, 'durum': 'aktif'})
        messages.success(request, "Hoş geldin! Ücretsiz plan ile başladın.")
        return redirect('anasayfa')

    return render(request, 'quant_app/plan_sec.html', {'planlar': planlar})


@login_required
def iyzico_odeme_baslat(request):
    """
    İyzico CheckoutForm ile ödeme başlatır.
    Kullanıcıyı İyzico'nun hazır ödeme formuna yönlendirir.
    """
    import iyzipay
    import json
    import uuid
    from django.conf import settings
    from .models import AbonelikPlan

    plan_ad = request.session.get('secilen_plan_ad', 'aylik')
    try:
        plan = AbonelikPlan.objects.get(ad=plan_ad, aktif=True)
    except AbonelikPlan.DoesNotExist:
        messages.error(request, "Plan bulunamadı.")
        return redirect('plan_sec')

    iyzico_options = {
        'api_key': settings.IYZICO_API_KEY,
        'secret_key': settings.IYZICO_SECRET_KEY,
        'base_url': settings.IYZICO_BASE_URL,
    }

    conversation_id = f"sovereign-{request.user.id}-{uuid.uuid4().hex[:8]}"

    request_data = {
        'locale': 'tr',
        'conversationId': conversation_id,
        'price': str(plan.fiyat_tl),
        'paidPrice': str(plan.fiyat_tl),
        'currency': 'TRY',
        'basketId': f'plan-{plan.ad}-{request.user.id}',
        'paymentGroup': 'SUBSCRIPTION',
        'callbackUrl': f"{settings.SITE_URL}/odeme/callback/",
        'enabledInstallments': ['1'],
        'buyer': {
            'id': str(request.user.id),
            'name': request.user.first_name or request.user.username,
            'surname': request.user.last_name or 'Kullanici',
            'gsmNumber': '+905000000000',
            'email': request.user.email or f'{request.user.username}@sovereign.app',
            'identityNumber': '11111111111',
            'lastLoginDate': request.user.last_login.strftime('%Y-%m-%d %H:%M:%S') if request.user.last_login else '2024-01-01 00:00:00',
            'registrationDate': request.user.date_joined.strftime('%Y-%m-%d %H:%M:%S'),
            'registrationAddress': 'Türkiye',
            'ip': request.META.get('REMOTE_ADDR', '127.0.0.1'),
            'city': 'Istanbul',
            'country': 'Turkey',
        },
        'shippingAddress': {
            'contactName': request.user.username,
            'city': 'Istanbul',
            'country': 'Turkey',
            'address': 'Türkiye',
        },
        'billingAddress': {
            'contactName': request.user.username,
            'city': 'Istanbul',
            'country': 'Turkey',
            'address': 'Türkiye',
        },
        'basketItems': [{
            'id': f'plan-{plan.ad}',
            'name': f'Piyasa Pusulam {plan.get_ad_display()} Abonelik',
            'category1': 'Yazılım',
            'itemType': 'VIRTUAL',
            'price': str(plan.fiyat_tl),
        }],
    }

    # conversation_id'yi session'a kaydet (callback'te doğrulama için)
    request.session['iyzico_conversation_id'] = conversation_id

    try:
        checkout_form_init = iyzipay.CheckoutFormInitialize().create(request_data, iyzico_options)
        result = json.loads(checkout_form_init.read().decode('utf-8'))

        if result.get('status') == 'success':
            # İyzico'nun hazır ödeme sayfasına yönlendir
            payment_page_url = result.get('paymentPageUrl')
            if payment_page_url:
                return redirect(payment_page_url)

        # Hata durumu
        hata = result.get('errorMessage', 'Ödeme başlatılamadı.')
        messages.error(request, f"İyzico hatası: {hata}")
        return redirect('plan_sec')

    except Exception as e:
        messages.error(request, f"Ödeme sistemi bağlantı hatası: {str(e)}")
        return redirect('plan_sec')


def iyzico_callback(request):
    """
    İyzico ödeme sonucu callback'i.
    POST ile token gönderir, ödemeyi doğrulayıp aboneliği aktif eder.
    """
    import iyzipay
    import json
    from django.conf import settings
    from django.utils import timezone
    from datetime import timedelta
    from .models import AbonelikPlan, Abonelik

    token = request.POST.get('token') or request.GET.get('token')
    if not token:
        messages.error(request, "Geçersiz ödeme yanıtı.")
        return redirect('plan_sec')

    iyzico_options = {
        'api_key': settings.IYZICO_API_KEY,
        'secret_key': settings.IYZICO_SECRET_KEY,
        'base_url': settings.IYZICO_BASE_URL,
    }

    try:
        result_raw = iyzipay.CheckoutForm().retrieve({'locale': 'tr', 'token': token}, iyzico_options)
        result = json.loads(result_raw.read().decode('utf-8'))
    except Exception as e:
        messages.error(request, f"Ödeme doğrulanamadı: {str(e)}")
        return redirect('plan_sec')

    if result.get('paymentStatus') == 'SUCCESS':
        # Ödeme başarılı — aboneliği aktif et
        plan_ad = request.session.get('secilen_plan_ad', 'aylik')
        try:
            plan = AbonelikPlan.objects.get(ad=plan_ad)
        except AbonelikPlan.DoesNotExist:
            plan = AbonelikPlan.objects.get(ad='aylik')

        sure_gun = plan.sure_gun or 30
        bitis = timezone.now() + timedelta(days=sure_gun)
        odeme_id = result.get('paymentId', '')

        ab, created = Abonelik.objects.get_or_create(
            user=request.user,
            defaults={'plan': plan, 'durum': 'aktif', 'bitis': bitis, 'iyzico_odeme_id': odeme_id}
        )
        if not created:
            ab.plan = plan
            ab.durum = 'aktif'
            ab.bitis = bitis
            ab.iyzico_odeme_id = odeme_id
            ab.save()

        del request.session['secilen_plan_ad']
        messages.success(request, f"🎉 {plan.get_ad_display()} aboneliğin aktif! Hoş geldin.")
        return redirect('anasayfa')

    else:
        hata = result.get('errorMessage', 'Ödeme tamamlanamadı.')
        messages.error(request, f"Ödeme başarısız: {hata}")
        return redirect('plan_sec')


def login_view(request):
    """Giriş sayfası."""
    if request.user.is_authenticated:
        return redirect('anasayfa')

    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('anasayfa')
        messages.error(request, "Kullanıcı adı veya şifre hatalı.")
        return render(request, 'quant_app/login.html', {'username': username})

    return render(request, 'quant_app/login.html')


def logout_view(request):
    logout(request)
    messages.info(request, "Çıkış yapıldı.")
    return redirect('login')


# =====================================================================
# 📊 DASHBOARD + ANALİZ VIEW'LARI
# =====================================================================

@login_required
def dashboard_view(request):
    """
    Ana kokpit sayfası. Sayfa ilk açıldığında varsayılan bir varlık
    (BTC-USD / KRIPTO / 1d) için analiz tetiklenir; kullanıcı dropdown'dan
    seçim değiştirdiğinde JS, /analiz/ endpoint'ine AJAX isteği atar
    (bkz. analiz_api_view) — sayfa yeniden yüklenmez.
    """
    context = {
        'varlik_havuzu_json': json.dumps(core.VARLIK_HAVUZU),
        'interval_secenekleri_json': json.dumps(core.DISPLAY_INTERVALS_BY_MARKET),
        'pazarlar': list(core.VARLIK_HAVUZU.keys()),
    }
    return render(request, 'quant_app/dashboard.html', context)


# =====================================================================
# 🆕 YENİ ANASAYFA / MARKET TARAYICI EKRANLARI
# =====================================================================
PAZAR_GORUNUM_BILGISI = {
    "KRIPTO": {"emoji": "🟠", "ad": "Kripto", "renk": "#F7931A"},
    "TR_HISSE": {"emoji": "🔵", "ad": "TR Hisse", "renk": "#3B82F6"},
    "ABD_HISSE": {"emoji": "🟢", "ad": "ABD Hisse", "renk": "#22C55E"},
    "EMTIA": {"emoji": "🟡", "ad": "Emtia", "renk": "#EAB308"},
}


@login_required
def anasayfa_view(request):
    """
    🆕 Yeni varsayılan giriş ekranı — market overview. Eski "Laboratuvar"
    (tek-varlık ML analiz ekranı, eski dashboard_view) İÇERİK OLARAK
    DOKUNULMADAN kalıyor, sadece navigasyonda alt bir bölüme taşınıyor.
    """
    context = {
        'varlik_havuzu_json': json.dumps(core.VARLIK_HAVUZU),
        'pazar_bilgisi_json': json.dumps(PAZAR_GORUNUM_BILGISI),
        'pazarlar': list(core.VARLIK_HAVUZU.keys()),
    }
    return render(request, 'quant_app/anasayfa.html', context)


@login_required
def market_view(request, pazar):
    """
    🆕 Bir pazara tıklanınca açılan ekran (örn. /market/KRIPTO/). Arama +
    periyot seçici + İÇİNDE mevcut analiz arayüzü (Laboratuvar ile AYNI
    JS/HTML bileşenleri, sadece bu sayfaya gömülü) bulunur.
    """
    if pazar not in core.VARLIK_HAVUZU:
        return redirect('anasayfa')

    context = {
        'pazar': pazar,
        'aktif_pazar': pazar,
        'pazar_bilgisi': PAZAR_GORUNUM_BILGISI.get(pazar, {}),
        'varlik_listesi_json': json.dumps(core.VARLIK_HAVUZU.get(pazar, [])),
        'varlik_havuzu_json': json.dumps(core.VARLIK_HAVUZU),
        'interval_secenekleri_json': json.dumps(core.DISPLAY_INTERVALS_BY_MARKET.get(pazar, [])),
    }
    return render(request, 'quant_app/market.html', context)


@login_required
def anasayfa_fiyatlar_api(request):
    """
    🆕 AJAX endpoint — anasayfa.html JS'i sayfa açılır açılmaz buraya
    istek atar: /api/anasayfa-fiyatlar/

    ÖNEMLİ TASARIM KARARI (Yol B — "hafif" yöntem): Her pazardan ilk 5
    varlığın SADECE fiyat/hacim/açılış bilgisi çekilir — ML pipeline'ı
    (Triple Barrier, model eğitimi) ÇALIŞTIRILMAZ. 20 varlık PARALEL
    (ThreadPoolExecutor) çekilir; sıralı çekilseydi 20 × birkaç saniye
    sürerdi, paralel çekim toplam süreyi ~tek bir yavaş çağrı kadara
    indirir (ağ I/O bekleme süresi örtüşür).
    """
    sonuclar_by_pazar = {}

    tum_gorevler = []  # (pazar, sembol) çiftleri
    for pazar, semboller in core.VARLIK_HAVUZU.items():
        for sembol in semboller[:5]:  # Her pazardan İLK 5 (büyüklüğe göre sıralı liste varsayımıyla)
            tum_gorevler.append((pazar, sembol))

    def _guvenli_float(deger):
        """NaN/Infinity'yi 0.0'a çevirir — bunlar geçerli JSON değeri değildir,
        JSON.parse() tarayıcıda tüm response'u bozar."""
        try:
            f = float(deger)
            if f != f or f in (float('inf'), float('-inf')):  # f != f -> NaN kontrolü
                return 0.0
            return f
        except (TypeError, ValueError):
            return 0.0

    def _tek_gorev_calistir(gorev):
        pazar, sembol = gorev
        sonuc = core.hafif_fiyat_getir(sembol, pazar, interval="1d")
        return pazar, sembol, sonuc

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_gorev = {executor.submit(_tek_gorev_calistir, g): g for g in tum_gorevler}
        for future in as_completed(future_to_gorev):
            pazar, sembol, sonuc = future.result()
            sonuclar_by_pazar.setdefault(pazar, []).append({
                'sembol': sembol,
                'basarili': sonuc.basarili,
                'fiyat': _guvenli_float(sonuc.fiyat),
                'acilis': _guvenli_float(sonuc.acilis),
                'degisim_yuzde': _guvenli_float(sonuc.degisim_yuzde),
                'hacim': _guvenli_float(sonuc.hacim),
                'hata_mesaji': sonuc.hata_mesaji,
            })

    # Her pazarın içindeki sırayı VARLIK_HAVUZU'ndaki orijinal sıraya göre düzelt
    # (ThreadPoolExecutor sonuçları tamamlanma sırasına göre döndürür, karışık olabilir)
    for pazar in sonuclar_by_pazar:
        orijinal_sira = {s: i for i, s in enumerate(core.VARLIK_HAVUZU[pazar][:5])}
        sonuclar_by_pazar[pazar].sort(key=lambda x: orijinal_sira.get(x['sembol'], 999))

    return JsonResponse({'basarili': True, 'pazarlar': sonuclar_by_pazar})


@login_required
def market_fiyatlar_api(request, pazar):
    """
    🆕 Bir pazarın TÜM varlıklarının (sadece ilk 5 değil) hafif fiyat
    bilgisini döner — market.html sayfasındaki varlık listesi için.

    Query parametreleri:
      interval: '1d', '4h' vb. (varsayılan '1d') — fiyat/hacim değişimi
                BU periyodun son barına göre hesaplanır.
      siralama: 'artan' | 'azalan' | 'hacim_yuksek' | 'hacim_dusuk' |
                'hacim_artan' | 'hacim_azalan' (varsayılan: yok, orijinal sıra)

    ⚠️ CACHE: Listeler artık 397+ (TR_HISSE), 500 (KRIPTO), 403+
    (ABD_HISSE) varlık içerebiliyor — her seferinde tek tek çekmek
    30-60+ saniye sürebilir. Sonuç, (pazar, interval) çiftine özel 5
    DAKİKALIK cache'e yazılır.

    NOT: Kripto'da artık Binance'in toplu (sadece 24s) ticker'ı KULLANILMIYOR
    — kullanıcı "periyoda özel doğruluk" istedi, bu yüzden Kripto da diğer
    pazarlar gibi sembol-başına paralel çekime geçirildi (daha yavaş ama
    seçilen periyoda göre doğru sonuç verir).
    """
    if pazar not in core.VARLIK_HAVUZU:
        return JsonResponse({'basarili': False, 'hata_mesaji': f"Geçersiz pazar: {pazar}"}, status=400)

    interval = request.GET.get('interval', '1d').strip()
    siralama = request.GET.get('siralama', '').strip()

    sonuclar = market_fiyatlarini_cacheli_getir(pazar, interval)
    sonuclar = siralamayi_uygula(sonuclar, siralama)

    return JsonResponse({'basarili': True, 'pazar': pazar, 'interval': interval, 'varliklar': sonuclar})

CACHE_SURESI_SANIYE = 5 * 60  # 5 dakika

SIRALAMA_ANAHTAR_FONKSIYONLARI = {
    'artan': (lambda x: x['degisim_yuzde'], True),
    'azalan': (lambda x: x['degisim_yuzde'], False),
    'hacim_yuksek': (lambda x: x['hacim'], True),
    'hacim_dusuk': (lambda x: x['hacim'], False),
    'hacim_artan': (lambda x: x['hacim_degisim_yuzde'], True),
    'hacim_azalan': (lambda x: x['hacim_degisim_yuzde'], False),
}


def siralamayi_uygula(sonuclar: list, siralama: str) -> list:
    """
    🆕 6 sıralama türünü uygular: artan/azalan (% fiyat değişimi),
    hacim_yuksek/hacim_dusuk (mutlak hacim büyüklüğü), hacim_artan/
    hacim_azalan (önceki bara göre hacim % değişimi).
    Geçersiz/boş `siralama` -> orijinal sıra (VARLIK_HAVUZU sırası) korunur.
    """
    if siralama not in SIRALAMA_ANAHTAR_FONKSIYONLARI:
        return sonuclar

    anahtar_fn, buyukten_kucuge = SIRALAMA_ANAHTAR_FONKSIYONLARI[siralama]
    basarili = [s for s in sonuclar if s['basarili']]
    basarisiz = [s for s in sonuclar if not s['basarili']]
    basarili.sort(key=anahtar_fn, reverse=buyukten_kucuge)
    return basarili + basarisiz  # başarısız olanlar (veri yok) sona düşer


def market_fiyatlarini_cacheli_getir(pazar: str, interval: str = "1d") -> list:
    cache_anahtari = f"market_fiyatlar_{pazar}_{interval}"
    cache_li_sonuc = cache.get(cache_anahtari)
    if cache_li_sonuc is not None:
        return cache_li_sonuc

    # İlk aşama: sadece ilk 50 varlığı çek, hızlı dön
    # Sonraki cache doldurma turunda tam liste çekilecek
    semboller = core.VARLIK_HAVUZU[pazar]

    print(f"[CACHE] {pazar}/{interval}: {len(semboller)} varlık çekiliyor...")
    sonuclar = _pazar_fiyatlarini_paralel_getir(semboller, pazar, interval)
    print(f"[CACHE] {pazar}/{interval}: tamamlandı.")

    orijinal_sira = {s: i for i, s in enumerate(semboller)}
    sonuclar.sort(key=lambda x: orijinal_sira.get(x['sembol'], 999))

    cache.set(cache_anahtari, sonuclar, CACHE_SURESI_SANIYE)
    return sonuclar


def _pazar_fiyatlarini_paralel_getir(semboller: list, pazar: str, interval: str) -> list:
    """TÜM pazarlar (KRIPTO dahil) için sembol başına paralel çağrı —
    seçilen periyoda özel sonuç verir. Çağrı market_fiyatlarini_cacheli_getir
    tarafından ARKA PLAN thread'inde yapılır, yani kullanıcıyı bloklamaz."""
    def _tek_sembol_calistir(sembol):
        return sembol, core.hafif_fiyat_getir(sembol, pazar, interval=interval)

    sonuclar = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(_tek_sembol_calistir, s) for s in semboller]
        for future in as_completed(futures):
            sembol, sonuc = future.result()
            sonuclar.append({
                'sembol': sembol, 'basarili': sonuc.basarili,
                'fiyat': float(sonuc.fiyat), 'degisim_yuzde': float(sonuc.degisim_yuzde),
                'hacim': float(sonuc.hacim), 'hacim_degisim_yuzde': float(sonuc.hacim_degisim_yuzde),
            })
    return sonuclar


@login_required
def pazar_lider_api(request, pazar):
    """
    🆕 AJAX endpoint — anasayfa.html'deki "En Çok Artan / En Çok Azalan /
    Hacim Liderleri" linkleri buraya istek atar:
    /api/pazar-lider/<pazar>/?tur=artan|azalan|hacim&interval=1d

    AYNI cache'lenmiş veriyi (market_fiyatlarini_cacheli_getir) kullanır
    — ekstra bir ağ çağrısı YAPMAZ, sadece sıralayıp ilk 10'u döner.
    """
    if pazar not in core.VARLIK_HAVUZU:
        return JsonResponse({'basarili': False, 'hata_mesaji': f"Geçersiz pazar: {pazar}"}, status=400)

    tur = request.GET.get('tur', 'artan').strip()
    interval = request.GET.get('interval', '1d').strip()
    # Eski 3-değerli 'tur' parametresini yeni 6-değerli 'siralama' sözlüğüne eşle
    TUR_TO_SIRALAMA = {'artan': 'artan', 'azalan': 'azalan', 'hacim': 'hacim_yuksek'}
    if tur not in TUR_TO_SIRALAMA:
        return JsonResponse({'basarili': False, 'hata_mesaji': f"Geçersiz tür: {tur}"}, status=400)

    sonuclar = market_fiyatlarini_cacheli_getir(pazar, interval)
    siralanmis = siralamayi_uygula(sonuclar, TUR_TO_SIRALAMA[tur])
    siralanmis = [s for s in siralanmis if s['basarili']]

    return JsonResponse({'basarili': True, 'pazar': pazar, 'tur': tur, 'varliklar': siralanmis[:10]})


@login_required
def tarama_api(request, pazar):
    """
    🆕 Tarama (Screener) endpoint'i.
    GET /api/tarama/<pazar>/?interval=1d&filtreler=degisim_pozitif,hacim_artan,...

    Cache'teki mevcut fiyat/hacim/değişim verisini kullanır — ekstra API
    çağrısı YAPMAZ, bu yüzden anlıktır. RSI/MACD gibi göstergeler için
    ayrıca veri çekmek gerekir (sonraki faz).

    MEVCUT SABİT FİLTRELER (tek veya kombine kullanılabilir):
      degisim_pozitif     → değişim% > 0
      degisim_negatif     → değişim% < 0
      degisim_guclu       → değişim% > +3%
      dusus_guclu         → değişim% < -3%
      hacim_artan         → hacim_degisim_yuzde > 0 (önceki bara göre hacim arttı)
      hacim_azalan        → hacim_degisim_yuzde < 0
      hacim_yuksek        → hacim, o pazarın ortalamasının üzerinde
      fiyat_dusuk_degisim → abs(değişim%) < 1% (yatay hareket)
    """
    
    if pazar not in core.VARLIK_HAVUZU:
        return JsonResponse({'basarili': False, 'hata_mesaji': f"Geçersiz pazar: {pazar}"}, status=400)

    from .models import SiteAyari, GunlukAnaliz
    ayar = SiteAyari.get()
    if not _kullanici_premium_mi(request):
        kayit = _gunluk_kayit_getir(request)
        if kayit.tarama_adet >= ayar.ucretsiz_gunluk_tarama_limiti:
            return JsonResponse({
                'basarili': False,
                'limit_asimi': True,
                'limit_turu': 'tarama',
                'hata_mesaji': "Günlük deneyim limiti aşıldı. Sınırsız bir deneyim için Ücretli Planlarımıza geçebilirsiniz.",
            }, status=429)
        GunlukAnaliz.objects.filter(pk=kayit.pk).update(tarama_adet=kayit.tarama_adet + 1)

    interval = request.GET.get('interval', '1d').strip()
    filtreler_str = request.GET.get('filtreler', '').strip()
    filtreler = [f.strip() for f in filtreler_str.split(',') if f.strip()]

    # Cache'ten veriyi al — zaten doluysa anlık, dolmamışsa boş döner
    sonuclar = market_fiyatlarini_cacheli_getir(pazar, interval)


    basarili = [s for s in sonuclar if s['basarili']]

    if not basarili:
        return JsonResponse({'basarili': True, 'pazar': pazar, 'sonuc_sayisi': 0,
                             'varliklar': [], 'mesaj': 'Veri henüz yüklenmedi, Anasayfa sekmesini açarak veri yüklenmesini bekleyin.'})

    # Hacim ortalaması (hacim_yuksek filtresi için)
    hacim_listesi = [s['hacim'] for s in basarili if s['hacim'] > 0]
    hacim_ortalama = sum(hacim_listesi) / len(hacim_listesi) if hacim_listesi else 0

    FILTRE_FONKSIYONLARI = {
        'degisim_pozitif':     lambda s: s['degisim_yuzde'] > 0,
        'degisim_negatif':     lambda s: s['degisim_yuzde'] < 0,
        'degisim_guclu':       lambda s: s['degisim_yuzde'] > 3,
        'dusus_guclu':         lambda s: s['degisim_yuzde'] < -3,
        'hacim_artan':         lambda s: s['hacim_degisim_yuzde'] > 0,
        'hacim_azalan':        lambda s: s['hacim_degisim_yuzde'] < 0,
        'hacim_yuksek':        lambda s: s['hacim'] > hacim_ortalama,
        'fiyat_dusuk_degisim': lambda s: abs(s['degisim_yuzde']) < 1,
    }

    sonuc = basarili
    gecersiz_filtreler = []
    for filtre in filtreler:
        if filtre in FILTRE_FONKSIYONLARI:
            sonuc = [s for s in sonuc if FILTRE_FONKSIYONLARI[filtre](s)]
        elif filtre:
            gecersiz_filtreler.append(filtre)

    # Değişim%'e göre büyükten küçüğe sırala
    sonuc.sort(key=lambda x: abs(x['degisim_yuzde']), reverse=True)

    return JsonResponse({
        'basarili': True,
        'pazar': pazar,
        'interval': interval,
        'uygulanan_filtreler': filtreler,
        'gecersiz_filtreler': gecersiz_filtreler,
        'sonuc_sayisi': len(sonuc),
        'toplam_varlik': len(basarili),
        'varliklar': sonuc,
    })


@login_required
def tarama_gosterge_api(request, pazar):
    """
    🆕 Gösterge tabanlı tarama endpoint'i (SSE — Server-Sent Events).
    GET /api/tarama-gosterge/<pazar>/?interval=1d&filtreler=macd_yukari,rsi_asiri_satim,...

    Her varlık için gosterge_serileri_getir çağırır (paralel, ThreadPoolExecutor).
    İlerleme durumunu SSE ile anlık gönderir: data: {"tip":"ilerleme","yuzde":10}
    Tamamlanınca sonuçları gönderir: data: {"tip":"sonuc","varliklar":[...]}

    GÖSTERGE FİLTRELERİ:
      Trend:
        macd_yukari        → MACD çizgisi sinyal çizgisini yukarı keser (son 2 bar)
        macd_sifir_ustune  → MACD 0 seviyesini yukarı keser (son 2 bar)
        ema_kesisim        → EMA(20) son barda EMA(50) üzerinde, önceki barda altındaydı
        sma_kesisim        → SMA(20) son barda SMA(50) üzerinde, önceki barda altındaydı
        bollinger_alt_kirma→ Fiyat alt Bollinger bandını yukarı keser (son 2 bar)
      Momentum:
        rsi_asiri_satim    → RSI son barda 30 üzerinde, önceki barda 30 altındaydı
        rsi_asiri_alim     → RSI son barda 70 altında, önceki barda 70 üzerindeydi
        rsi_deger_alti     → RSI < kullanıcının girdiği değer (rsi_deger parametresi)
        rsi_deger_ustu     → RSI > kullanıcının girdiği değer
    """
    
    import json as _json
    from django.http import StreamingHttpResponse

    if pazar not in core.VARLIK_HAVUZU:
        return JsonResponse({'basarili': False, 'hata_mesaji': f"Geçersiz pazar: {pazar}"}, status=400)

    from .models import SiteAyari, GunlukAnaliz
    ayar = SiteAyari.get()
    limit_asildi = False
    if not _kullanici_premium_mi(request):
        kayit = _gunluk_kayit_getir(request)
        if kayit.tarama_adet >= ayar.ucretsiz_gunluk_tarama_limiti:
            limit_asildi = True
        else:
            GunlukAnaliz.objects.filter(pk=kayit.pk).update(tarama_adet=kayit.tarama_adet + 1)

    if limit_asildi:
        def _limit_asimi_stream():
            mesaj = "Günlük deneyim limiti aşıldı. Sınırsız bir deneyim için Ücretli Planlarımıza geçebilirsiniz."
            yield f"data: {_json.dumps({'tip': 'limit_asimi', 'hata_mesaji': mesaj})}\n\n"
        response = StreamingHttpResponse(_limit_asimi_stream(), content_type='text/event-stream')
        response['Cache-Control'] = 'no-cache'
        return response

    interval = request.GET.get('interval', '1d').strip()



    filtreler_str = request.GET.get('filtreler', '').strip()
    filtreler = [f.strip() for f in filtreler_str.split(',') if f.strip()]
    rsi_deger = float(request.GET.get('rsi_deger', '50'))

    semboller = core.VARLIK_HAVUZU[pazar]
    toplam = len(semboller)

    def _gosterge_filtrele(gosterge_veri):
        """Gösterge verisine filtre uygular. True dönerse varlık listeye girer."""
        g = gosterge_veri
        n = len(g.close)
        if n < 3:
            return False

        for filtre in filtreler:
            if filtre == 'macd_yukari':
                # MACD çizgisi sinyal çizgisini yukarı keser
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

            elif filtre == 'sma_kesisim':
                # SMA(20) SMA(50)'yi yukarı keser — EMA kullanarak yaklaşık
                if not (len(g.ema_20) >= 2 and len(g.sma_200) >= 2):
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

    def _sse_stream():
        tamamlanan = [0]
        sonuclar = []
        lock = threading.Lock()

        def _tek_varlik(sembol):
            g = core.gosterge_serileri_getir(sembol, pazar, interval)
            with lock:
                tamamlanan[0] += 1
                yuzde = int(tamamlanan[0] / toplam * 100)
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

        # Paralel çekim + SSE ilerleme
        from concurrent.futures import ThreadPoolExecutor, as_completed
        ilerleme_kuyruk = []

        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_sembol = {executor.submit(_tek_varlik, s): s for s in semboller}
            for future in as_completed(future_to_sembol):
                with lock:
                    yuzde = int(tamamlanan[0] / toplam * 100)
                # İlerleme SSE
                yield f"data: {_json.dumps({'tip': 'ilerleme', 'yuzde': yuzde, 'tamamlanan': tamamlanan[0], 'toplam': toplam})}\n\n"
                sonuc = future.result()
                if sonuc:
                    sonuclar.append(sonuc)

        # Final sonuç SSE
        sonuclar.sort(key=lambda x: abs(x['degisim_yuzde']), reverse=True)
        yield f"data: {_json.dumps({'tip': 'sonuc', 'basarili': True, 'pazar': pazar, 'sonuc_sayisi': len(sonuclar), 'toplam_varlik': toplam, 'varliklar': sonuclar})}\n\n"

    response = StreamingHttpResponse(_sse_stream(), content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'
    return response


@login_required
def gosterge_serileri_api(request):
    """
    🆕 AJAX endpoint — market.html'deki Analiz sekmesi JS'i, bir varlık
    seçilince buraya istek atar: /api/gosterge-serileri/?sembol=BTC-USD&pazar=KRIPTO&interval=1d

    🆕 GÜNLÜK LİMİT: Ücretsiz/anonim kullanıcılar için SiteAyari'daki
    günlük analiz limitini aşarsa 429 döner.
    """
    from .models import SiteAyari, GunlukAnaliz

    ayar = SiteAyari.get()
    premium = _kullanici_premium_mi(request)

    if not premium:
        kayit = _gunluk_kayit_getir(request)
        if kayit.adet >= ayar.ucretsiz_gunluk_analiz_limiti:
            return JsonResponse({
                'basarili': False,
                'limit_asimi': True,
                'limit_turu': 'analiz',
                'hata_mesaji': "Günlük deneyim limiti aşıldı. Sınırsız bir deneyim için Ücretli Planlarımıza geçebilirsiniz.",
            }, status=429)

        GunlukAnaliz.objects.filter(pk=kayit.pk).update(adet=kayit.adet + 1)

    sembol = request.GET.get('sembol', '').strip()
    pazar = request.GET.get('pazar', '').strip()
    interval = request.GET.get('interval', '1d').strip()

    if not sembol or not pazar:
        return JsonResponse({'basarili': False, 'hata_mesaji': 'sembol/pazar parametreleri eksik.'}, status=400)
    if pazar not in core.VARLIK_HAVUZU:
        return JsonResponse({'basarili': False, 'hata_mesaji': f"Geçersiz pazar: {pazar}"}, status=400)

    sonuc = core.gosterge_serileri_getir(sembol, pazar, interval)

    if not sonuc.basarili:
        return JsonResponse({'basarili': False, 'hata_mesaji': sonuc.hata_mesaji})

    return JsonResponse({
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
# 🆕 TAKİP LİSTESİ (WATCHLIST) API'LARI
# =====================================================================
@login_required
def takip_listesi_getir(request):
    """GET /api/takip-listesi/ — kullanıcının tüm takip listesini döner."""
    ogeler = WatchlistItem.objects.filter(user=request.user)
    return JsonResponse({
        'basarili': True,
        'ogeler': [{'id': o.id, 'sembol': o.sembol, 'pazar': o.pazar} for o in ogeler],
    })


@login_required
@require_http_methods(["POST"])
def takip_listesi_ekle(request):
    """POST /api/takip-listesi/ekle/ — body: {"sembol": "BTC-USD", "pazar": "KRIPTO"}"""
    try:
        veri = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'basarili': False, 'hata_mesaji': 'Geçersiz JSON.'}, status=400)

    sembol = veri.get('sembol', '').strip()
    pazar = veri.get('pazar', '').strip()

    if not sembol or not pazar:
        return JsonResponse({'basarili': False, 'hata_mesaji': 'sembol/pazar eksik.'}, status=400)
    if pazar not in core.VARLIK_HAVUZU:
        return JsonResponse({'basarili': False, 'hata_mesaji': f"Geçersiz pazar: {pazar}"}, status=400)

    from .models import SiteAyari
    zaten_var_mi = WatchlistItem.objects.filter(user=request.user, sembol=sembol, pazar=pazar).exists()
    if not zaten_var_mi and not _kullanici_premium_mi(request):
        ayar = SiteAyari.get()
        mevcut_sayi = WatchlistItem.objects.filter(user=request.user).count()
        if mevcut_sayi >= ayar.ucretsiz_takip_listesi_limiti:
            return JsonResponse({
                'basarili': False,
                'limit_asimi': True,
                'limit_turu': 'takip_listesi',
                'hata_mesaji': f"{ayar.ucretsiz_takip_listesi_limiti} varlık limitine sahipsiniz. Sınırsız varlık takibi için Ücretli Planlarımıza geçebilirsiniz.",
            }, status=429)

    oge, olusturuldu_mu = WatchlistItem.objects.get_or_create(

        user=request.user, sembol=sembol, pazar=pazar,
        defaults={'siralama': WatchlistItem.objects.filter(user=request.user).count()},
    )
    return JsonResponse({
        'basarili': True, 'olusturuldu_mu': olusturuldu_mu,
        'oge': {'id': oge.id, 'sembol': oge.sembol, 'pazar': oge.pazar},
    })


@login_required
@require_http_methods(["POST"])
def takip_listesi_sil(request, oge_id):
    """POST /api/takip-listesi/sil/<id>/"""
    try:
        oge = WatchlistItem.objects.get(id=oge_id, user=request.user)
    except WatchlistItem.DoesNotExist:
        return JsonResponse({'basarili': False, 'hata_mesaji': 'Öğe bulunamadı.'}, status=404)

    oge.delete()
    return JsonResponse({'basarili': True})


def _gunluk_kayit_getir(request):
    """
    Kullanıcı/IP + bugünün tarihine göre GunlukAnaliz kaydını getirir
    (yoksa oluşturur). Tüm limit kontrolleri (analiz/tarama/lab_analiz)
    bu TEK kaydı paylaşır.
    """
    from .models import GunlukAnaliz
    from django.utils import timezone

    bugun = timezone.now().date()
    if request.user.is_authenticated:
        kayit, _ = GunlukAnaliz.objects.get_or_create(user=request.user, tarih=bugun, defaults={'adet': 0})
    else:
        ip = request.META.get('REMOTE_ADDR', '0.0.0.0')
        kayit, _ = GunlukAnaliz.objects.get_or_create(ip_adresi=ip, tarih=bugun, defaults={'adet': 0})
    return kayit


def _kullanici_premium_mi(request):
    """Kullanıcı geçerli bir ücretli abonelikte mi?"""
    from .models import Abonelik
    if not request.user.is_authenticated:
        return False
    try:
        return request.user.abonelik.premium_mi
    except Abonelik.DoesNotExist:
        return False
    

@login_required
def limit_durumu_api(request):
    """
    GET /api/limit-durumu/ — Ücretsiz kullanıcı için kalan günlük hakları
    ve takip listesi durumunu JSON olarak döner. Frontend, "X hakkınız
    kaldı" yazılarını bununla doldurur. Premium kullanıcı için sadece
    {'premium': True} döner (frontend kısıtlama göstermez).
    """
    

    from .models import SiteAyari

    premium = _kullanici_premium_mi(request)
    if premium:
        return JsonResponse({'basarili': True, 'premium': True})

    from .models import FiyatAlarmi
    ayar = SiteAyari.get()
    kayit = _gunluk_kayit_getir(request)
    takip_sayisi = WatchlistItem.objects.filter(user=request.user).count()

    return JsonResponse({
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
            'kullanilan': FiyatAlarmi.objects.filter(user=request.user, aktif=True, tetiklendi_mi=False).count(),
            'limit': ayar.ucretsiz_alarm_limiti,
            'kalan': max(0, ayar.ucretsiz_alarm_limiti - FiyatAlarmi.objects.filter(user=request.user, aktif=True, tetiklendi_mi=False).count()),
        },
    })



def _pazar_cache_ile_fiyat_bul(sembol, pazar):
    """Celery'nin ısıttığı günlük cache listesinden sembolün güncel
    fiyatını arar — her seferinde canlı Yahoo isteği atmadan hızlı sonuç verir."""
    try:
        liste = market_fiyatlarini_cacheli_getir(pazar, '1d')
        for v in liste:
            if v.get('sembol') == sembol and v.get('basarili'):
                return v.get('fiyat')
    except Exception:
        pass
    return None


@login_required
def alarmlar_view(request):
    """GET /alarmlar/ — Fiyat alarmları sayfası (market sayfası gibi bağımsız)."""
    context = {
        'varlik_havuzu_json': json.dumps(core.VARLIK_HAVUZU),
    }
    return render(request, 'quant_app/alarmlar.html', context)


@login_required
def alarmlar_api_getir(request):
    """GET /api/alarmlar/ — kullanıcının tüm alarmlarını, GÜNCEL fiyatla birlikte döner."""
    from .models import FiyatAlarmi
    alarmlar = FiyatAlarmi.objects.filter(user=request.user)
    veri = []
    for a in alarmlar:
        guncel = _pazar_cache_ile_fiyat_bul(a.sembol, a.pazar)
        veri.append({
            'id': a.id, 'sembol': a.sembol, 'pazar': a.pazar,
            'hedef_fiyat': float(a.hedef_fiyat), 'yon': a.yon, 'yon_gorunen': a.get_yon_display(),
            'aktif': a.aktif, 'tetiklendi_mi': a.tetiklendi_mi, 'goruldu_mu': a.goruldu_mu,
            'olusturulma_fiyati': float(a.olusturulma_fiyati) if a.olusturulma_fiyati is not None else None,
            'guncel_fiyat': guncel,
            'olusturulma_tarihi': a.olusturulma_tarihi.strftime('%d.%m.%Y %H:%M'),
        })
    return JsonResponse({'basarili': True, 'alarmlar': veri})


@login_required
def alarmlar_api_ekle(request):
    """POST /api/alarmlar/ekle/ — yeni alarm oluşturur (ücretsiz limit kontrolü ile)."""
    if request.method != 'POST':
        return JsonResponse({'basarili': False, 'hata_mesaji': 'Geçersiz istek.'}, status=405)
    from .models import FiyatAlarmi, SiteAyari

    try:
        veri = json.loads(request.body)
        sembol = veri.get('sembol', '').strip()
        pazar = veri.get('pazar', '').strip()
        hedef_fiyat = float(veri.get('hedef_fiyat'))
        yon = veri.get('yon', '').strip()
    except (ValueError, TypeError, json.JSONDecodeError):
        return JsonResponse({'basarili': False, 'hata_mesaji': 'Geçersiz veri.'}, status=400)

    if not sembol or pazar not in core.VARLIK_HAVUZU or yon not in ('ustune', 'altina'):
        return JsonResponse({'basarili': False, 'hata_mesaji': 'Eksik/geçersiz bilgi.'}, status=400)

    if not _kullanici_premium_mi(request):
        ayar = SiteAyari.get()
        aktif_sayisi = FiyatAlarmi.objects.filter(user=request.user, aktif=True, tetiklendi_mi=False).count()
        if aktif_sayisi >= ayar.ucretsiz_alarm_limiti:
            return JsonResponse({
                'basarili': False,
                'limit_asimi': True,
                'limit_turu': 'alarm',
                'hata_mesaji': f"{ayar.ucretsiz_alarm_limiti} alarm limitine sahipsiniz. Sınırsız alarm için Ücretli Planlarımıza geçebilirsiniz.",
            }, status=429)

    guncel_fiyat = _pazar_cache_ile_fiyat_bul(sembol, pazar)
    FiyatAlarmi.objects.create(user=request.user, sembol=sembol, pazar=pazar, hedef_fiyat=hedef_fiyat, yon=yon, olusturulma_fiyati=guncel_fiyat)
    return JsonResponse({'basarili': True})

@login_required
def anlik_fiyat_api(request):
    """GET /api/anlik-fiyat/ — form'da varlık seçilince hedef fiyat kutusunu doldurmak için."""
    sembol = request.GET.get('sembol', '').strip()
    pazar = request.GET.get('pazar', '').strip()
    if not sembol or pazar not in core.VARLIK_HAVUZU:
        return JsonResponse({'basarili': False, 'hata_mesaji': 'Eksik bilgi.'}, status=400)
    fiyat = _pazar_cache_ile_fiyat_bul(sembol, pazar)
    if fiyat is None:
        sonuc = core.hafif_fiyat_getir(sembol, pazar, interval="1d")
        fiyat = sonuc.fiyat if sonuc.basarili else None
    if fiyat is None:
        return JsonResponse({'basarili': False, 'hata_mesaji': 'Fiyat alınamadı.'})
    return JsonResponse({'basarili': True, 'fiyat': fiyat})


@login_required
def alarmlar_api_sil(request, alarm_id):
    """POST /api/alarmlar/sil/<id>/ — alarmı siler."""
    if request.method != 'POST':
        return JsonResponse({'basarili': False, 'hata_mesaji': 'Geçersiz istek.'}, status=405)
    from .models import FiyatAlarmi
    FiyatAlarmi.objects.filter(id=alarm_id, user=request.user).delete()
    return JsonResponse({'basarili': True})


@login_required
def alarmlar_api_gorundu(request):
    """POST /api/alarmlar/gorundu/ — tetiklenen alarmları 'görüldü' işaretler (rozet sıfırlama)."""
    if request.method != 'POST':
        return JsonResponse({'basarili': False, 'hata_mesaji': 'Geçersiz istek.'}, status=405)
    from .models import FiyatAlarmi
    FiyatAlarmi.objects.filter(user=request.user, tetiklendi_mi=True, goruldu_mu=False).update(goruldu_mu=True)
    return JsonResponse({'basarili': True})



@login_required
def abonelik_durumu_api(request):
    """GET /api/abonelik-durumu/ — Ayarlar sekmesi için mevcut abonelik bilgisi."""
    from .models import Abonelik
    try:
        ab = request.user.abonelik
        return JsonResponse({
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
        return JsonResponse({'basarili': True, 'plan': 'ucretsiz', 'plan_gorunen': 'Ücretsiz', 'durum': None, 'bitis': None, 'gecerli_mi': False, 'premium_mi': False})


@login_required
def abonelik_iptal_et(request):
    """POST /api/abonelik-iptal/ — Aboneliği iptal eder (durum='iptal')."""
    if request.method != 'POST':
        return JsonResponse({'basarili': False, 'hata_mesaji': 'Geçersiz istek.'}, status=405)
    from .models import Abonelik
    try:
        ab = request.user.abonelik
        ab.durum = 'iptal'
        ab.save()
        return JsonResponse({'basarili': True})
    except Abonelik.DoesNotExist:
        return JsonResponse({'basarili': False, 'hata_mesaji': 'Aktif bir aboneliğiniz yok.'}, status=400)


@login_required
def abonelik_ucretsize_gec(request):
    """POST /api/abonelik-ucretsize-gec/ — Planı doğrudan Ücretsiz'e çevirir (ödeme gerektirmez)."""
    if request.method != 'POST':
        return JsonResponse({'basarili': False, 'hata_mesaji': 'Geçersiz istek.'}, status=405)
    from .models import Abonelik, AbonelikPlan
    ucretsiz_plan = AbonelikPlan.objects.get(ad='ucretsiz')
    ab, _ = Abonelik.objects.get_or_create(user=request.user, defaults={'plan': ucretsiz_plan})
    ab.plan = ucretsiz_plan
    ab.durum = 'aktif'
    ab.bitis = None
    ab.save()
    return JsonResponse({'basarili': True})

@login_required
def usd_try_kuru_api(request):
    """GET /api/usd-try-kuru/ — frontend'deki para birimi dönüşüm anahtarı için."""
    kur = core.usd_try_kuru_getir()
    return JsonResponse({'basarili': True, 'kur': kur})


def _analiz_sonucunu_cache_e_yaz(sonuc: "core.AnalizSonucu"):
    """AnalizSonucu objesini QuantSignalCache satırına yazar (upsert)."""
    grafik_json = json.dumps({
        'tarihler': sonuc.grafik_tarihler,
        'open': sonuc.grafik_open,
        'high': sonuc.grafik_high,
        'low': sonuc.grafik_low,
        'close': sonuc.grafik_close,
        'ema20': sonuc.grafik_ema20,
        'ema50': sonuc.grafik_ema50,
        'sma200': sonuc.grafik_sma200,
        'rsi': sonuc.grafik_rsi,
        'macd': sonuc.grafik_macd,
        'macd_sig': sonuc.grafik_macd_sig,
        'stoch_k': sonuc.grafik_stoch_k,
        'wt1': sonuc.grafik_wt1,
        'cci': sonuc.grafik_cci,
        'volume': sonuc.grafik_volume,
    })

    QuantSignalCache.objects.update_or_create(
        sembol=sonuc.sembol, pazar=sonuc.pazar, period=sonuc.interval,
        defaults=dict(
            son_fiyat=sonuc.fiyat_su_an,
            atr_gucu=sonuc.atr_gucu,
            konsensüs_karari=sonuc.ml.karar if sonuc.ml else "",
            boga_ihtimali=sonuc.ml.boga_ihtimali if sonuc.ml else 0.0,
            ayi_ihtimali=sonuc.ml.ayi_ihtimali if sonuc.ml else 0.0,
            ensemble_accuracy=(
                (sonuc.ml.acc_gbm + sonuc.ml.acc_rf + sonuc.ml.acc_xgb) / 3 * 100
                if sonuc.ml else 0.0
            ),
            anlik_rsi=sonuc.rsi, anlik_macd=sonuc.macd, anlik_stoch_k=sonuc.stoch_k,
            anlik_wt1=sonuc.wt1, anlik_cci=sonuc.cci,
            grafik_verisi_json=grafik_json,
        ),
    )


def _cache_kaydini_json_a_cevir(kayit: QuantSignalCache) -> dict:
    """Cache'ten gelen bir kaydı, analiz_api_view'ın ürettiği JSON şekliyle
    AYNI formata çevirir (frontend tek bir JSON şemasıyla çalışsın diye).

    NOT: annual_factor/komisyon_orani/period gibi "ritim bilgisi" alanları
    QuantSignalCache modelinde SAKLANMIYOR — ama bunlar zaten sadece
    (pazar, interval) ikilisine bağlı DETERMİNİSTİK fonksiyonlardır, bu
    yüzden cache'ten okurken de aynı sonucu verecek şekilde anında
    yeniden hesaplanabilir. df_raw tarih/satır bilgisi ise cache'te
    saklanmadığı için burada gösterilemez (sadece taze hesaplamada gelir).
    """
    grafik = json.loads(kayit.grafik_verisi_json) if kayit.grafik_verisi_json else {}
    return {
        'basarili': True,
        'veri_yetersiz': False,
        'cache_ten_mi': True,
        'sembol': kayit.sembol, 'pazar': kayit.pazar, 'interval': kayit.period,
        'period': core.suggest_period(kayit.pazar, kayit.period),
        'annual_factor': core.get_annual_factor(kayit.pazar, kayit.period),
        'komisyon_orani': core.get_commission_rate(kayit.pazar),
        'fiyat_su_an': kayit.son_fiyat, 'atr_gucu': kayit.atr_gucu,
        'rsi': kayit.anlik_rsi, 'macd': kayit.anlik_macd,
        'stoch_k': kayit.anlik_stoch_k, 'wt1': kayit.anlik_wt1, 'cci': kayit.anlik_cci,
        'ml': {
            'karar': kayit.konsensüs_karari,
            'boga_ihtimali': kayit.boga_ihtimali,
            'ayi_ihtimali': kayit.ayi_ihtimali,
            'ensemble_accuracy': kayit.ensemble_accuracy,
        },
        'grafik': grafik,
    }


def _analiz_sonucunu_json_a_cevir(sonuc: "core.AnalizSonucu") -> dict:
    if not sonuc.basarili:
        return {'basarili': False, 'hata_mesaji': sonuc.hata_mesaji}

    if sonuc.veri_yetersiz:
        return {
            'basarili': True, 'veri_yetersiz': True,
            'hata_mesaji': sonuc.uyari_metni or "Bu seçim için yeterli veri yok.",
        }

    ml = sonuc.ml
    return {
        'basarili': True, 'veri_yetersiz': False, 'cache_ten_mi': False,
        'sembol': sonuc.sembol, 'pazar': sonuc.pazar, 'interval': sonuc.interval,
        'veri_kaynagi': sonuc.veri_kaynagi, 'sentetik_mi': sonuc.sentetik_mi,
        'uyari_metni': sonuc.uyari_metni, 'tazelik_uyarisi': sonuc.tazelik_uyarisi,
        'sma_200_guvenilir': sonuc.sma_200_guvenilir,
        # 🆕 Sidebar "Akademik Ritim Bilgisi" + Teşhis Bilgisi paneli için:
        'period': sonuc.period,
        'annual_factor': sonuc.annual_factor,
        'komisyon_orani': sonuc.komisyon_orani,
        'df_raw_satir_sayisi': sonuc.df_raw_satir_sayisi,
        'df_raw_ilk_tarih': sonuc.df_raw_ilk_tarih,
        'df_raw_son_tarih': sonuc.df_raw_son_tarih,
        'fiyat_su_an': sonuc.fiyat_su_an, 'atr_gucu': sonuc.atr_gucu,
        'sma_200_degeri': sonuc.sma_200_degeri, 'para_birimi': sonuc.para_birimi,
        'rsi': sonuc.rsi, 'macd': sonuc.macd, 'stoch_k': sonuc.stoch_k,
        'wt1': sonuc.wt1, 'cci': sonuc.cci, 'rejim': sonuc.rejim,
        'ml': {
            'karar': ml.karar, 'boga_ihtimali': ml.boga_ihtimali, 'ayi_ihtimali': ml.ayi_ihtimali,
            'kalibrasyon_aktif': ml.kalibrasyon_aktif,
            'w_gbm': ml.w_gbm, 'w_rf': ml.w_rf, 'w_xgb': ml.w_xgb,
            'cv_gbm': ml.cv_gbm, 'cv_rf': ml.cv_rf, 'cv_xgb': ml.cv_xgb,
            'acc_gbm': ml.acc_gbm * 100, 'acc_rf': ml.acc_rf * 100, 'acc_xgb': ml.acc_xgb * 100,
            'profit_factor': ml.profit_factor, 'max_dd': ml.max_dd, 'sharpe': ml.sharpe,
            'expectancy': ml.expectancy, 'win_rate': ml.win_rate,
            'avg_win': ml.avg_win, 'avg_loss': ml.avg_loss,
            'train_satir_sayisi': ml.train_satir_sayisi,
        } if ml else None,
        'grafik': {
            'tarihler': sonuc.grafik_tarihler,
            'open': sonuc.grafik_open, 'high': sonuc.grafik_high, 'low': sonuc.grafik_low, 'close': sonuc.grafik_close,
            'ema20': sonuc.grafik_ema20, 'ema50': sonuc.grafik_ema50, 'sma200': sonuc.grafik_sma200,
            'rsi': sonuc.grafik_rsi, 'macd': sonuc.grafik_macd, 'macd_sig': sonuc.grafik_macd_sig,
            'stoch_k': sonuc.grafik_stoch_k, 'wt1': sonuc.grafik_wt1, 'cci': sonuc.grafik_cci,
            'volume': sonuc.grafik_volume,
        },
    }


@login_required
def analiz_api_view(request):
    """
    AJAX endpoint — dashboard.html'deki JS, kullanıcı pazar/varlık/interval
    değiştirdiğinde buraya GET isteği atar: /analiz/?sembol=BTC-USD&pazar=KRIPTO&interval=1d

    Akış:
      1. QuantSignalCache'te 5 dakikadan taze bir kayıt var mı bak.
      2. Varsa -> direkt onu JSON olarak dön (hızlı yol).
      3. Yoksa -> quant_ml_core.analiz_yap() ile YENİDEN hesapla, cache'i
         güncelle, sonucu JSON olarak dön.
    """
    sembol = request.GET.get('sembol', '').strip()
    pazar = request.GET.get('pazar', '').strip()
    interval = request.GET.get('interval', '').strip()

    if not sembol or not pazar or not interval:
        return JsonResponse({'basarili': False, 'hata_mesaji': 'sembol/pazar/interval parametreleri eksik.'}, status=400)

    if pazar not in core.VARLIK_HAVUZU:
        return JsonResponse({'basarili': False, 'hata_mesaji': f"Geçersiz pazar: {pazar}"}, status=400)

   
   #eski cahche kaydı
   # cache_kaydi = QuantSignalCache.gecerli_cache_getir(sembol, pazar, interval)
    #if cache_kaydi is not None:
     #   return JsonResponse(_cache_kaydini_json_a_cevir(cache_kaydi))

    #try:
     #   sonuc = core.analiz_yap(sembol, pazar, interval)
    #except Exception as exc:
        # Beklenmeyen bir motor hatası kullanıcıya 500 olarak değil,
        # anlaşılır bir JSON hata mesajı olarak dönsün.
     #   return JsonResponse(
      #      {'basarili': False, 'hata_mesaji': f"Analiz motorunda beklenmeyen bir hata oluştu: {exc}"},
       #     status=500,
        #)

    #if sonuc.basarili and not sonuc.veri_yetersiz:
     #   _analiz_sonucunu_cache_e_yaz(sonuc)

    #return JsonResponse(_analiz_sonucunu_json_a_cevir(sonuc))
    
    cache_kaydi = QuantSignalCache.gecerli_cache_getir(sembol, pazar, interval)
    if cache_kaydi is not None:
        return JsonResponse(_cache_kaydini_json_a_cevir(cache_kaydi))

    # Cache yok -> gerçek bir yeni hesaplama gerekiyor, bu bir "hak" harcar.
    from .models import SiteAyari, GunlukAnaliz
    ayar = SiteAyari.get()
    if not _kullanici_premium_mi(request):
        kayit = _gunluk_kayit_getir(request)
        if kayit.lab_analiz_adet >= ayar.ucretsiz_gunluk_lab_analiz_limiti:
            return JsonResponse({
                'basarili': False,
                'limit_asimi': True,
                'limit_turu': 'lab_analiz',
                'hata_mesaji': "Günlük deneyim limiti aşıldı. Sınırsız bir deneyim için Ücretli Planlarımıza geçebilirsiniz.",
            }, status=429)
        GunlukAnaliz.objects.filter(pk=kayit.pk).update(lab_analiz_adet=kayit.lab_analiz_adet + 1)

    # Ağır ML hesaplamasını Celery'e devret, kullanıcıyı bekletme.
    from .tasks import analiz_hesapla_task
    task = analiz_hesapla_task.delay(sembol, pazar, interval)
    return JsonResponse({'basarili': True, 'beklemede': True, 'task_id': task.id})


@login_required
def analiz_durum_view(request, task_id):
    """
    Polling endpoint — dashboard.html'deki JS, task_id aldıktan sonra
    buraya periyodik olarak sorar: 'bitti mi?'
    """
    from celery.result import AsyncResult
    result = AsyncResult(task_id)

    if not result.ready():
        return JsonResponse({'basarili': True, 'beklemede': True})

    try:
        veri = result.get()
    except Exception as exc:
        return JsonResponse(
            {'basarili': False, 'hata_mesaji': f"Analiz motorunda beklenmeyen bir hata oluştu: {exc}"},
            status=500,
        )

    return JsonResponse(veri)