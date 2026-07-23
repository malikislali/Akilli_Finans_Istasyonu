"""
quant_app/api_serializers.py — 🆕 Mobil app için DRF serializer'ları.

Web tarafındaki register_view (views.py) DOKUNULMADAN duruyor.
Bu serializer aynı User modelini kullanıyor, sadece mobil app'in JSON
gönderip JSON alması için bir "çevirmen" katmanı.
"""
from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from rest_framework import serializers

from .models import Abonelik


class KayitSerializer(serializers.ModelSerializer):
    """Mobil app'ten kayıt için. Web'deki register_view ile aynı
    validasyon mantığı (şifre eşleşmesi, min uzunluk vb.)."""
    password = serializers.CharField(write_only=True, min_length=6, validators=[validate_password])
    password2 = serializers.CharField(write_only=True, min_length=6)

    class Meta:
        model = User
        fields = ('username', 'email', 'password', 'password2')

    def validate(self, data):
        if data['password'] != data['password2']:
            raise serializers.ValidationError({'password2': 'Şifreler eşleşmiyor.'})
        if User.objects.filter(username=data['username']).exists():
            raise serializers.ValidationError({'username': 'Bu kullanıcı adı zaten alınmış.'})
        return data

    def create(self, validated_data):
        validated_data.pop('password2')
        user = User.objects.create_user(
            username=validated_data['username'],
            email=validated_data.get('email', ''),
            password=validated_data['password'],
        )
        return user


class KullaniciBilgiSerializer(serializers.ModelSerializer):
    """Giriş yapmış kullanıcının /api/v1/auth/me/ ile göreceği bilgi.
    Abonelik durumu da buraya gömülü — mobile app her açılışta tek
    istekle hem kullanıcıyı hem plan durumunu öğrenebilsin diye."""
    abonelik_plani = serializers.SerializerMethodField()
    abonelik_gecerli_mi = serializers.SerializerMethodField()
    abonelik_bitis = serializers.SerializerMethodField()

    class Meta:
        model = User
        fields = ('id', 'username', 'email', 'date_joined',
                   'abonelik_plani', 'abonelik_gecerli_mi', 'abonelik_bitis')

    def _abonelik(self, obj):
        return getattr(obj, 'abonelik', None)

    def get_abonelik_plani(self, obj):
        ab = self._abonelik(obj)
        return ab.plan.ad if ab else 'ucretsiz'

    def get_abonelik_gecerli_mi(self, obj):
        ab = self._abonelik(obj)
        return ab.gecerli_mi if ab else False

    def get_abonelik_bitis(self, obj):
        ab = self._abonelik(obj)
        return ab.bitis if ab and ab.bitis else None
