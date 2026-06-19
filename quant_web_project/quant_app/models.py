from django.db import models
from django.contrib.auth.models import User

# =====================================================================
# 💸 1. TABLO: PORTFÖY VE MANUEL İŞLEM GÜNLÜĞÜ MODELİ
# =====================================================================
class Portfolio(models.Model):
    """
    Kullanıcıların hangi finansal varlıktan, hangi maliyetle kaç adet 
    aldığını tutan ve anlık PnL (Kâr/Zarar) hesaplamasına zemin hazırlayan tablo.
    """
    PAZAR_CHOICES = [
        ('KRIPTO', 'Kripto Para'),
        ('TR_HISSE', 'Borsa İstanbul'),
        ('ABD_HISSE', 'ABD Borsaları'),
        ('EMTIA', 'Emtia / Değerli Maden'),
    ]

    # Bir kullanıcının birden fazla cüzdan kalemi olabilir (1-to-Many)
    user = models.ForeignKey(User, on_on_delete=models.CASCADE, related_name='portfolios', verbose_name="Yatırımcı")
    pazar = models.CharField(max_length=20, choices=PAZAR_CHOICES, verbose_name="Pazar Alanı")
    sembol = models.CharField(max_length=20, verbose_name="Varlık Sembolü (Örn: BTC-USD)")
    adet = models.DecimalField(max_length=18, decimal_places=6, verbose_name="Alınan Adet / Kontrat")
    alis_fiyati = models.DecimalField(max_length=18, decimal_places=4, verbose_name="Giriş / Alış Fiyatı")
    eklenme_tarihi = models.DateTimeField(auto_now_add=True, verbose_name="İşlem Zamanı")

    class Meta:
        verbose_name = "Portföy Kalemi"
        verbose_name_plural = "Portföy Cüzdanları"
        ordering = ['-eklenme_tarihi']

    def __str__(self):
        return f"{self.user.username} | {self.sembol} - {self.adet} Adet"

    @property
    def toplam_maliyet(self):
        """Kullanıcının bu pozisyona bağladığı toplam ana para hoca"""
        return self.adet * self.alis_fiyati


# =====================================================================
# 📡 2. TABLO: YAPAY ZEKA SİNYAL ÖNBELLEK (CACHE) MODELİ
# =====================================================================
class QuantSignalCache(models.Model):
    """
    YPMorgan disipliniyle sunucu performansını koruma zırhı!
    Yapay zekanın en son ürettiği tahmin rasyolarını ve grafik verilerini 
    burada saklarız, böylece sayfa her açıldığında yfinance'i beklemeyiz hoca.
    """
    sembol = models.CharField(max_length=20, unique=True, verbose_name="Varlık Sembolü")
    pazar = models.CharField(max_length=20, verbose_name="Pazar")
    period = models.CharField(max_length=10, verbose_name="Mum Periyodu")
    
    # Canlı Veri Seti Kesiti
    son_fiyat = models.FloatField(verbose_name="Son Güncel Fiyat")
    degisim_24s = models.FloatField(verbose_name="Periyot Değişimi %")
    atr_gucu = models.FloatField(verbose_name="Mevcut ATR Volatilitesi")
    
    # Yapay Zeka Jüri Kararı
    konsensüs_karari = models.CharField(max_length=20, verbose_name="Kurul Kararı (ARTIŞ/AZALIŞ)")
    boga_ihtimali = models.FloatField(verbose_name="Boğa Olasılığı %")
    ayi_ihtimali = models.FloatField(verbose_name="Ayı Olasılığı %")
    ensemble_accuracy = models.FloatField(verbose_name="Model Tarihsel Başarısı %")
    
    # 🧠 Plotly'nin jet gibi çizmesi için grafik listelerini JSON olarak saklıyoruz hoca
    grafik_verisi_json = models.TextField(verbose_name="Grafik Listeleri JSON Küpü")
    
    son_guncellenme = models.DateTimeField(auto_now=True, verbose_name="Son Sinyal Üretim Zamanı")

    class Meta:
        verbose_name = "Yapay Zeka Sinyal Havuzu"
        verbose_name_plural = "Yapay Zeka Sinyal Havuzları"

    def __str__(self):
        return f"{self.sembol} | {self.konsensüs_karari} (%{self.boga_ihtimali:.1f} Boğa)"