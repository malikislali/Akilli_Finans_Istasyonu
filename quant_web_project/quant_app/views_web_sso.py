"""
quant_app/views_web_sso.py — 🆕 Adım 20

Adım 20'nin web (session) tarafı. `api_views_web_sso.py`'deki
MobilWebGirisTokenView'ın ürettiği tek kullanımlık token'ı burada
karşılıyoruz — DİKKAT: bu JWT DEĞİL, düz bir Django view, çünkü amacı
tam olarak SESSION (çerez tabanlı) oturum açmak; web sitesinin geri
kalanı zaten session auth kullanıyor, buna dokunmuyoruz.
"""
from django.contrib.auth import login as django_login
from django.contrib.auth.models import User
from django.core.cache import cache
from django.shortcuts import redirect

from .api_views_web_sso import MOBIL_GIRIS_CACHE_ANAHTAR_ONEKI


def mobil_web_giris(request, token):
    """GET /mobil-giris/<token>/?next=/plan-sec/
    Mobil app'ten gelen tek kullanımlık token'ı doğrular, geçerliyse
    Django session login yapar ve `next` parametresindeki sayfaya
    yönlendirir. Token geçersiz/süresi dolmuşsa normal giriş sayfasına
    düşer (kullanıcı elle giriş yapabilir)."""
    cache_anahtari = f'{MOBIL_GIRIS_CACHE_ANAHTAR_ONEKI}{token}'
    kullanici_id = cache.get(cache_anahtari)

    if not kullanici_id:
        return redirect('login')

    # Tek kullanımlık — hemen sil, aynı token ikinci kez işe yaramasın
    cache.delete(cache_anahtari)

    try:
        kullanici = User.objects.get(pk=kullanici_id)
    except User.DoesNotExist:
        return redirect('login')

    django_login(request, kullanici, backend='django.contrib.auth.backends.ModelBackend')

    hedef = request.GET.get('next', '/plan-sec/')
    # Güvenlik: sadece SİTE İÇİ göreli yollara izin ver (open redirect önleme)
    if not hedef.startswith('/'):
        hedef = '/plan-sec/'
    return redirect(hedef)
