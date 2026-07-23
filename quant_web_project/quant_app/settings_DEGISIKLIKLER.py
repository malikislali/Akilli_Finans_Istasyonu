"""
settings.py DEĞİŞİKLİK REHBERİ
================================
Bu dosya kopyala-yapıştır değil — mevcut settings.py'ndaki ilgili
bölümlere aşağıdaki eklemeleri yapman için bir rehber. Mevcut hiçbir
satırı SİLME, sadece aşağıdakileri EKLE.

Web tarafı (session auth, template render, CSRF middleware) olduğu
gibi kalıyor. JWT sadece /api/v1/ altındaki yeni endpoint'ler için
devreye giriyor — birbirini etkilemiyorlar.
"""

# ─────────────────────────────────────────────────────────────
# 1) INSTALLED_APPS içine EKLE (en altına, 'quant_app'tan sonra)
# ─────────────────────────────────────────────────────────────
"""
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'quant_app',
    'rest_framework',                          # 🆕
    'rest_framework_simplejwt',                # 🆕
    'rest_framework_simplejwt.token_blacklist', # 🆕 — BLACKLIST_AFTER_ROTATION için ŞART, yoksa refresh rotation hata verir
    'corsheaders',                              # 🆕 (mobil app farklı origin'den istek atacağı için)
]
"""

# ⚠️ ÖNEMLİ: token_blacklist app'i eklendikten sonra migration çalıştırman gerekiyor:
#   python manage.py migrate token_blacklist

# ─────────────────────────────────────────────────────────────
# 2) MIDDLEWARE içine EKLE
#    CorsMiddleware, CommonMiddleware'den ÖNCE gelmeli (Django CORS
#    dokümantasyonunun şart koştuğu sıralama budur)
# ─────────────────────────────────────────────────────────────
"""
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'corsheaders.middleware.CorsMiddleware',           # 🆕 — CommonMiddleware'den ÖNCE
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]
"""

# ─────────────────────────────────────────────────────────────
# 3) Dosyanın SONUNA (SESSION_ ayarlarından sonra) EKLE
# ─────────────────────────────────────────────────────────────

from datetime import timedelta  # dosyanın başındaki import'lara da eklenebilir

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ),
    # NOT: DEFAULT_PERMISSION_CLASSES bilinçli olarak boş bırakıldı —
    # her view kendi permission'ını (IsAuthenticated / AllowAny) açıkça
    # belirtecek, "varsayılan güvenli" yaklaşımı yerine netlik tercih edildi.
    'DEFAULT_PERMISSION_CLASSES': (
        'rest_framework.permissions.IsAuthenticated',
    ),
}

SIMPLE_JWT = {
    # Access token kısa ömürlü: çalınsa bile hasar süresi sınırlı
    'ACCESS_TOKEN_LIFETIME': timedelta(minutes=30),
    # Refresh token uzun ömürlü: kullanıcı app'i her açtığında tekrar
    # giriş yapmasın diye. Mobile app AsyncStorage'da bunu saklayacak.
    'REFRESH_TOKEN_LIFETIME': timedelta(days=30),
    # Her refresh'te yeni bir refresh token da üretilir (rotation) —
    # eski refresh token'lar geçersiz olur, çalıntı token riskini azaltır
    'ROTATE_REFRESH_TOKENS': True,
    'BLACKLIST_AFTER_ROTATION': True,
    'AUTH_HEADER_TYPES': ('Bearer',),
    'USER_ID_FIELD': 'id',
    'USER_ID_CLAIM': 'user_id',
}

# Mobil app hangi origin'lerden istek atacak?
# Geliştirme sırasında Expo Metro bundler farklı portlardan servis eder,
# bu yüzden DEBUG modunda hepsine izin veriyoruz. Production'da
# (canlıya alırken) CORS_ALLOW_ALL_ORIGINS=True satırını KALDIR ve
# CORS_ALLOWED_ORIGINS'i gerçek app domain'inle sınırla — mobil native
# app'ler zaten "origin" göndermez, bu ayar asıl web-tabanlı (Expo Go /
# webview test) senaryolar için önemli.
if DEBUG:
    CORS_ALLOW_ALL_ORIGINS = True
else:
    CORS_ALLOWED_ORIGINS = os.environ.get('CORS_ALLOWED_ORIGINS', '').split(',') if os.environ.get('CORS_ALLOWED_ORIGINS') else []
