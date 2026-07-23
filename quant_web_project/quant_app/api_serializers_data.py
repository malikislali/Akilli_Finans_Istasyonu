"""
quant_app/api_serializers_data.py — GÜNCELLENDİ (Adım 3)
Bu dosyanın TAMAMINI mevcut api_serializers_data.py'nin yerine koy.

DEĞİŞİKLİK: WatchlistOgeSerializer'a guncel_fiyat ve degisim_yuzde
alanları eklendi (AlarmSerializer'daki guncel_fiyat mantığıyla aynı
yöntem — view içinde objeye enjekte edilip serializer'da okunuyor).
"""
from rest_framework import serializers
from .models import WatchlistItem, FiyatAlarmi


class WatchlistOgeSerializer(serializers.ModelSerializer):
    guncel_fiyat = serializers.SerializerMethodField()
    degisim_yuzde = serializers.SerializerMethodField()

    class Meta:
        model = WatchlistItem
        fields = ('id', 'sembol', 'pazar', 'eklenme_tarihi', 'guncel_fiyat', 'degisim_yuzde')
        read_only_fields = ('id', 'eklenme_tarihi')

    def get_guncel_fiyat(self, obj):
        return getattr(obj, '_guncel_fiyat', None)

    def get_degisim_yuzde(self, obj):
        return getattr(obj, '_degisim_yuzde', None)


class WatchlistEkleSerializer(serializers.Serializer):
    """Sadece giriş validasyonu için — model yerine düz Serializer,
    çünkü ekleme mantığı (limit kontrolü, get_or_create) view içinde
    web tarafındaki takip_listesi_ekle ile birebir aynı akışta kalıyor."""
    sembol = serializers.CharField(max_length=20)
    pazar = serializers.CharField(max_length=20)


class AlarmSerializer(serializers.ModelSerializer):
    yon_gorunen = serializers.CharField(source='get_yon_display', read_only=True)
    guncel_fiyat = serializers.SerializerMethodField()

    class Meta:
        model = FiyatAlarmi
        fields = (
            'id', 'sembol', 'pazar', 'hedef_fiyat', 'yon', 'yon_gorunen',
            'aktif', 'tetiklendi_mi', 'goruldu_mu', 'olusturulma_fiyati',
            'guncel_fiyat', 'olusturulma_tarihi',
        )
        read_only_fields = fields

    def get_guncel_fiyat(self, obj):
        return getattr(obj, '_guncel_fiyat', None)


class AlarmEkleSerializer(serializers.Serializer):
    sembol = serializers.CharField(max_length=20)
    pazar = serializers.CharField(max_length=20)
    hedef_fiyat = serializers.FloatField()
    yon = serializers.ChoiceField(choices=['ustune', 'altina'])
