"""
quant_app/api_views_auth.py — 🆕 Mobil app JWT auth endpoint'leri.

DİKKAT: Bu dosya views.py'a EKLENMİYOR, AYRI bir dosya. Web tarafının
login_view/register_view/logout_view fonksiyonları hiç değişmiyor.

Login/refresh için ayrıca view yazmıyoruz — simplejwt'nin hazır
TokenObtainPairView / TokenRefreshView'ı doğrudan api_urls.py'de
kullanılacak. Burada sadece simplejwt'nin sağlamadığı şeyler var:
kayıt, "ben kimim" bilgisi, çıkış (token blacklist).
"""
from rest_framework import status, generics, permissions
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.exceptions import TokenError
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer

from .api_serializers import KayitSerializer, KullaniciBilgiSerializer


class GirisSerializer(TokenObtainPairSerializer):
    """simplejwt'nin varsayılan login serializer'ı sadece access/refresh
    döner. Mobile app her girişte ayrıca /auth/me/'ye istek atmasın diye
    kullanıcı bilgisini de aynı response'a gömüyoruz."""

    def validate(self, attrs):
        data = super().validate(attrs)
        data['kullanici'] = KullaniciBilgiSerializer(self.user).data
        return data


class GirisView(TokenObtainPairView):
    """POST /api/v1/auth/login/
    Body: {username, password}
    Dönüş: {access, refresh, kullanici: {...}}"""
    serializer_class = GirisSerializer


class KayitView(generics.CreateAPIView):
    """POST /api/v1/auth/register/
    Body: {username, email, password, password2}
    Dönüş: 201 + access/refresh token (kayıt olur olmaz otomatik giriş)
    """
    permission_classes = [permissions.AllowAny]
    serializer_class = KayitSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()

        refresh = RefreshToken.for_user(user)
        return Response({
            'access': str(refresh.access_token),
            'refresh': str(refresh),
            'kullanici': KullaniciBilgiSerializer(user).data,
        }, status=status.HTTP_201_CREATED)


class BenKimimView(APIView):
    """GET /api/v1/auth/me/
    Header: Authorization: Bearer <access_token>
    App her açıldığında bu endpoint'e istek atıp hem kullanıcı bilgisini
    hem abonelik durumunu tek seferde öğrenir."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        return Response(KullaniciBilgiSerializer(request.user).data)


class CikisView(APIView):
    """POST /api/v1/auth/logout/
    Body: {refresh: "<refresh_token>"}
    Refresh token'ı blacklist'e alır — mobil app'te "çıkış yap"a
    basıldığında çağrılmalı, yoksa refresh token teorik olarak süresine
    kadar (30 gün) geçerli kalmaya devam eder."""
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        try:
            refresh_token = request.data.get('refresh')
            if not refresh_token:
                return Response({'hata': 'refresh alanı zorunlu.'}, status=status.HTTP_400_BAD_REQUEST)
            token = RefreshToken(refresh_token)
            token.blacklist()
            return Response(status=status.HTTP_205_RESET_CONTENT)
        except TokenError:
            return Response({'hata': 'Geçersiz token.'}, status=status.HTTP_400_BAD_REQUEST)
