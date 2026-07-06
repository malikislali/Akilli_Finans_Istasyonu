"""
models.py — Üyelik + sinyal önbelleği.

Mevcut Portfolio modeli AYNEN KORUNDU (bu aşamada kullanılmıyor ama
ileride "kendi pozisyonlarım" özelliği eklenince hazır olsun diye
silinmedi).

QuantSignalCache'e CACHE YARDIMCI METODLARI eklendi: Streamlit'teki
@st.cache_data(ttl=300) mantığının Django karşılığı. Kullanıcı bir
analiz istediğinde önce bu tabloya bakılır; son_guncellenme 5
dakikadan eskiyse yeniden hesaplanır (views.py içinde kullanılacak).
"""

from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from datetime import timedelta


class Portfolio(models.Model):
    PAZAR_CHOICES = [
        ('KRIPTO', 'Kripto Para'),
        ('TR_HISSE', 'Borsa İstanbul'),
        ('ABD_HISSE', 'ABD Borsaları'),
        ('EMTIA', 'Emtia / Değerli Maden'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='portfolios', verbose_name="Yatırımcı")
    pazar = models.CharField(max_length=20, choices=PAZAR_CHOICES, verbose_name="Pazar Alanı")
    sembol = models.CharField(max_length=20, verbose_name="Varlık Sembolü (Örn: BTC-USD)")
    adet = models.DecimalField(max_digits=18, decimal_places=6, verbose_name="Alınan Adet / Kontrat")
    alis_fiyati = models.DecimalField(max_digits=18, decimal_places=4, verbose_name="Giriş / Alış Fiyatı")
    eklenme_tarihi = models.DateTimeField(auto_now_add=True, verbose_name="İşlem Zamanı")

    class Meta:
        verbose_name = "Portföy Kalemi"
        verbose_name_plural = "Portföy Cüzdanları"
        ordering = ['-eklenme_tarihi']

    def __str__(self):
        return f"{self.user.username} | {self.sembol} - {self.adet} Adet"

    @property
    def toplam_maliyet(self):
        return self.adet * self.alis_fiyati


class QuantSignalCache(models.Model):
    """
    🆕 Django Faz 1 — basit TTL'li önbellek (Seçenek 2: anlık hesaplama
    + kısa süreli cache). Streamlit'teki `@st.cache_data(ttl=300)`
    davranışının doğrudan karşılığı.

    NOT: `sembol` alanı artık `unique=True` DEĞİL — aynı sembol farklı
    pazar/interval kombinasyonlarıyla birden çok kez analiz edilebilir
    (örn. BTC-USD hem "1d" hem "4h" için ayrı satır). Bu yüzden
    benzersizlik artık (sembol, pazar, period) üçlüsüne taşındı.
    """
    sembol = models.CharField(max_length=20, verbose_name="Varlık Sembolü")
    pazar = models.CharField(max_length=20, verbose_name="Pazar")
    period = models.CharField(max_length=10, verbose_name="Mum Periyodu (interval)")

    son_fiyat = models.FloatField(verbose_name="Son Güncel Fiyat")
    degisim_24s = models.FloatField(default=0.0, verbose_name="Periyot Değişimi %")
    atr_gucu = models.FloatField(verbose_name="Mevcut ATR Volatilitesi")

    konsensüs_karari = models.CharField(max_length=20, verbose_name="Kurul Kararı (ARTIŞ/AZALIŞ)")
    boga_ihtimali = models.FloatField(verbose_name="Boğa Olasılığı %")
    ayi_ihtimali = models.FloatField(verbose_name="Ayı Olasılığı %")
    ensemble_accuracy = models.FloatField(default=0.0, verbose_name="Model Tarihsel Başarısı %")

    anlik_rsi = models.FloatField(default=50.0, verbose_name="Anlık RSI Değeri")
    anlik_macd = models.FloatField(default=0.0, verbose_name="Anlık MACD Çizgisi")
    anlik_stoch_k = models.FloatField(default=50.0, verbose_name="Anlık Stochastic %K")
    anlik_wt1 = models.FloatField(default=0.0, verbose_name="Anlık WaveTrend WT1")
    anlik_cci = models.FloatField(default=0.0, verbose_name="Anlık CCI Değeri")

    grafik_verisi_json = models.TextField(blank=True, default="", verbose_name="Grafik Listeleri JSON Küpü")
    son_guncellenme = models.DateTimeField(auto_now=True, verbose_name="Son Sinyal Üretim Zamanı")

    class Meta:
        verbose_name = "Yapay Zeka Sinyal Havuzu"
        verbose_name_plural = "Yapay Zeka Sinyal Havuzları"
        unique_together = ('sembol', 'pazar', 'period')

    def __str__(self):
        return f"{self.sembol} ({self.period}) | {self.konsensüs_karari} | RSI: {self.anlik_rsi:.1f}"

    @property
    def tazelik_suresi_gecti_mi(self) -> bool:
        """5 dakikadan eski mi? (Streamlit ttl=300 karşılığı)."""
        return timezone.now() - self.son_guncellenme > timedelta(minutes=5)

    @classmethod
    def gecerli_cache_getir(cls, sembol: str, pazar: str, period: str):
        """
        Geçerli (5 dakikadan taze) bir cache kaydı varsa döner, yoksa
        None döner. views.py bunu çağırıp None gelirse yeniden hesaplar.
        """
        try:
            kayit = cls.objects.get(sembol=sembol, pazar=pazar, period=period)
        except cls.DoesNotExist:
            return None
        if kayit.tazelik_suresi_gecti_mi:
            return None
        return kayit


class WatchlistItem(models.Model):
    """
    🆕 Sol sidebar "Takip Listem" bölümü için kalıcı kayıt. Kullanıcı
    bir varlığı yıldızlayınca (⭐) buraya eklenir, ❌ ile silinir.

    NOT: Bu basit, TEK listelik bir yapı (proje notlarındaki "Sonraki
    Fazda: Çoklu takip listesi — Kripto Favorilerim / Uzun Vade / Trade
    Listem" önerisi henüz uygulanmadı). Çoklu liste eklenmek istenirse,
    buraya bir `liste_adi` alanı eklemek yeterli olur — model şu an
    buna kolayca genişletilebilir şekilde tasarlandı.
    """
    PAZAR_CHOICES = Portfolio.PAZAR_CHOICES

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='watchlist_items', verbose_name="Kullanıcı")
    pazar = models.CharField(max_length=20, choices=PAZAR_CHOICES, verbose_name="Pazar")
    sembol = models.CharField(max_length=20, verbose_name="Varlık Sembolü (Örn: BTC-USD)")
    eklenme_tarihi = models.DateTimeField(auto_now_add=True, verbose_name="Eklenme Zamanı")
    siralama = models.PositiveIntegerField(default=0, verbose_name="Görüntülenme Sırası")

    class Meta:
        verbose_name = "Takip Listesi Öğesi"
        verbose_name_plural = "Takip Listeleri"
        unique_together = ('user', 'pazar', 'sembol')  # Aynı kullanıcı aynı varlığı 2 kez ekleyemez
        ordering = ['siralama', 'eklenme_tarihi']

    def __str__(self):
        return f"{self.user.username} | ⭐ {self.sembol} ({self.pazar})"


class SiteAyari(models.Model):
    """
    Site geneli tek satırlık ayarlar (singleton pattern).
    Admin panelinden değiştirilebilir.
    """
    ucretli_kayit_aktif = models.BooleanField(
        default=False,
        verbose_name="Ücretli Kayıt Aktif",
        help_text="Kapalı: herkes ücretsiz kayıt olabilir. Açık: kayıtta plan seçimi ve ödeme istenir."
    )

    ucretsiz_gunluk_analiz_limiti = models.PositiveIntegerField(
        default=5,
        verbose_name="Ücretsiz Günlük Analiz Limiti (Market)",
        help_text="Giriş yapmamış veya ücretsiz plandaki kullanıcının günde kaç kez Market Analiz sekmesini kullanabileceği."
    )
    ucretsiz_gunluk_tarama_limiti = models.PositiveIntegerField(
        default=5,
        verbose_name="Ücretsiz Günlük Tarama Limiti",
        help_text="Ücretsiz kullanıcının günde kaç kez Tarama yapabileceği."
    )
    ucretsiz_gunluk_lab_analiz_limiti = models.PositiveIntegerField(
        default=3,
        verbose_name="Ücretsiz Günlük Laboratuvar Analiz Limiti",
        help_text="Ücretsiz kullanıcının günde kaç kez Laboratuvar'da ağır ML analizi çalıştırabileceği."
    )
    ucretsiz_takip_listesi_limiti = models.PositiveIntegerField(
        default=5,
        verbose_name="Ücretsiz Takip Listesi Limiti",
        help_text="Ücretsiz kullanıcının takip listesine ekleyebileceği maksimum varlık sayısı."
    )
    guncelleme_tarihi = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Site Ayarı"
        verbose_name_plural = "Site Ayarları"

    def __str__(self):
        durum = "ÜCRETLİ KAYIT AÇIK" if self.ucretli_kayit_aktif else "Ücretsiz Kayıt"
        return f"Site Ayarları — {durum}"

    @classmethod
    def get(cls):
        """Her zaman tek bir SiteAyari kaydı döner (yoksa oluşturur)."""
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj


class AbonelikPlan(models.Model):
    """Mevcut abonelik planları (Ücretsiz, Aylık, Yıllık vb.)"""
    AD_CHOICES = [
        ('ucretsiz', 'Ücretsiz'),
        ('aylik', 'Aylık'),
        ('yillik', 'Yıllık'),
    ]
    ad = models.CharField(max_length=20, choices=AD_CHOICES, unique=True, verbose_name="Plan Adı")
    aciklama = models.TextField(blank=True, verbose_name="Açıklama")
    fiyat_tl = models.DecimalField(max_digits=8, decimal_places=2, default=0, verbose_name="Fiyat (TL)")
    sure_gun = models.PositiveIntegerField(default=0, verbose_name="Süre (Gün, 0=Sınırsız)")
    aktif = models.BooleanField(default=True, verbose_name="Aktif")

    class Meta:
        verbose_name = "Abonelik Planı"
        verbose_name_plural = "Abonelik Planları"

    def __str__(self):
        return f"{self.get_ad_display()} — {self.fiyat_tl}₺"


class Abonelik(models.Model):
    """Kullanıcının aktif aboneliği."""
    DURUM_CHOICES = [
        ('aktif', 'Aktif'),
        ('pasif', 'Pasif'),
        ('deneme', 'Deneme'),
        ('iptal', 'İptal Edildi'),
    ]
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='abonelik', verbose_name="Kullanıcı")
    plan = models.ForeignKey(AbonelikPlan, on_delete=models.PROTECT, verbose_name="Plan")
    durum = models.CharField(max_length=10, choices=DURUM_CHOICES, default='aktif', verbose_name="Durum")
    baslangic = models.DateTimeField(auto_now_add=True, verbose_name="Başlangıç")
    bitis = models.DateTimeField(null=True, blank=True, verbose_name="Bitiş (null=Sınırsız)")
    iyzico_odeme_id = models.CharField(max_length=100, blank=True, verbose_name="İyzico Ödeme ID")
    notlar = models.TextField(blank=True, verbose_name="Admin Notları")

    class Meta:
        verbose_name = "Abonelik"
        verbose_name_plural = "Abonelikler"

    def __str__(self):
        return f"{self.user.username} — {self.plan} ({self.get_durum_display()})"

    @property
    def gecerli_mi(self):
        """Abonelik aktif ve süresi dolmamış mı?"""
        if self.durum != 'aktif':
            return False
        if self.bitis is None:
            return True
        return timezone.now() < self.bitis

    @property
    def premium_mi(self):
        """Ücretli bir plana sahip mi?"""
        return self.plan.ad in ('aylik', 'yillik') and self.gecerli_mi


class GunlukAnaliz(models.Model):
    """Ücretsiz kullanıcıların günlük analiz/tarama kullanımını takip eder."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True,
                              related_name='gunluk_analizler', verbose_name="Kullanıcı (null=anonim)")
    ip_adresi = models.GenericIPAddressField(null=True, blank=True, verbose_name="IP Adresi")
    tarih = models.DateField(auto_now_add=True, verbose_name="Tarih")
    adet = models.PositiveIntegerField(default=0, verbose_name="Analiz Sayısı (Market Analiz Sekmesi)")
    tarama_adet = models.PositiveIntegerField(default=0, verbose_name="Tarama Sayısı (Market Tarama Sekmesi)")
    lab_analiz_adet = models.PositiveIntegerField(default=0, verbose_name="Lab ML Analiz Sayısı")

 
    class Meta:
        verbose_name = "Günlük Analiz Kullanımı"
        verbose_name_plural = "Günlük Analiz Kullanımları"
        unique_together = [('user', 'tarih'), ('ip_adresi', 'tarih')]

    def __str__(self):
        kim = self.user.username if self.user else self.ip_adresi
        return f"{kim} — {self.tarih} — {self.adet} analiz"

