"""
admin.py — Sovereign Cockpit Admin Paneli

Özelleştirilmiş Django admin:
- Kullanıcı + Abonelik listesi (aynı ekranda)
- SiteAyari (ücretli kayıt anahtarı)
- AbonelikPlan yönetimi
- GunlukAnaliz istatistikleri
"""

from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import User
from django.utils.html import format_html
from django.utils.safestring import mark_safe
from django.utils import timezone
from .models import (
    SiteAyari, AbonelikPlan, Abonelik, GunlukAnaliz,
    WatchlistItem, QuantSignalCache
)

# ── Site başlığını özelleştir ──
admin.site.site_header = "🏛️ Sovereign Cockpit — Admin"
admin.site.site_title = "Sovereign Admin"
admin.site.index_title = "Yönetim Paneli"


# ── SiteAyari (singleton) ──
@admin.register(SiteAyari)
class SiteAyariAdmin(admin.ModelAdmin):
    list_display = ('__str__', 'ucretli_kayit_durumu', 'ucretsiz_gunluk_analiz_limiti', 'guncelleme_tarihi')
    readonly_fields = ('guncelleme_tarihi',)

    def ucretli_kayit_durumu(self, obj):
        if obj.ucretli_kayit_aktif:
            #return format_html('<span style="color:green;font-weight:bold;">🔓 AÇIK — Ücretli kayıt zorunlu</span>')
            return mark_safe('<span style="color:green;font-weight:bold;">🔓 AÇIK — Ücretli kayıt zorunlu</span>')

        return format_html('<span style="color:gray;">🔒 KAPALI — Ücretsiz kayıt</span>')
    ucretli_kayit_durumu.short_description = "Ücretli Kayıt"

    def has_add_permission(self, request):
        # Singleton — sadece 1 kayıt olabilir
        return not SiteAyari.objects.exists()

    def has_delete_permission(self, request, obj=None):
        return False


# ── AbonelikPlan ──
@admin.register(AbonelikPlan)
class AbonelikPlanAdmin(admin.ModelAdmin):
    list_display = ('get_ad_display', 'fiyat_tl', 'sure_gun', 'aktif', 'abone_sayisi')
    list_editable = ('aktif',)

    def abone_sayisi(self, obj):
        return obj.abonelik_set.filter(durum='aktif').count()
    abone_sayisi.short_description = "Aktif Abone"


# ── Abonelik ──
class AbonelikInline(admin.StackedInline):
    model = Abonelik
    extra = 0
    readonly_fields = ('baslangic', 'gecerli_mi_goster')
    fields = ('plan', 'durum', 'bitis', 'iyzico_odeme_id', 'notlar', 'baslangic', 'gecerli_mi_goster')

    def gecerli_mi_goster(self, obj):
        if obj.gecerli_mi:
            return format_html('<span style="color:green;">✅ Geçerli</span>')
        return format_html('<span style="color:red;">❌ Geçersiz/Süresi Dolmuş</span>')
    gecerli_mi_goster.short_description = "Durum"


@admin.register(Abonelik)
class AbonelikAdmin(admin.ModelAdmin):
    list_display = ('user', 'plan', 'durum_badge', 'baslangic', 'bitis', 'gecerli_mi_goster')
    list_filter = ('durum', 'plan')
    search_fields = ('user__username', 'user__email')
    readonly_fields = ('baslangic',)
    list_per_page = 30
    date_hierarchy = 'baslangic'
    actions = ['aktif_yap', 'pasif_yap']

    def durum_badge(self, obj):
        renkler = {'aktif': 'green', 'pasif': 'gray', 'deneme': 'orange', 'iptal': 'red'}
        renk = renkler.get(obj.durum, 'gray')
        return format_html('<span style="color:{};font-weight:bold;">{}</span>',
                           renk, obj.get_durum_display())
    durum_badge.short_description = "Durum"

    def gecerli_mi_goster(self, obj):
        if obj.gecerli_mi:
            return format_html('<span style="color:green;">✅</span>')
        return format_html('<span style="color:red;">❌</span>')
    gecerli_mi_goster.short_description = "Geçerli"

    @admin.action(description="Seçili abonelikleri AKTİF yap")
    def aktif_yap(self, request, queryset):
        queryset.update(durum='aktif')

    @admin.action(description="Seçili abonelikleri PASİF yap")
    def pasif_yap(self, request, queryset):
        queryset.update(durum='pasif')


# ── Kullanıcı listesini abonelik bilgisiyle genişlet ──
class SovereignUserAdmin(UserAdmin):
    list_display = ('username', 'email', 'date_joined', 'last_login', 'plan_badge', 'abonelik_durumu')
    inlines = [AbonelikInline]

    def plan_badge(self, obj):
        try:
            ab = obj.abonelik
            renkler = {'ucretsiz': 'gray', 'aylik': 'blue', 'yillik': 'purple'}
            renk = renkler.get(ab.plan.ad, 'gray')
            return format_html('<span style="color:{};font-weight:bold;">{}</span>',
                               renk, ab.plan.get_ad_display())
        except Abonelik.DoesNotExist:
            return format_html('<span style="color:gray;">—</span>')
    plan_badge.short_description = "Plan"

    def abonelik_durumu(self, obj):
        try:
            ab = obj.abonelik
            if ab.gecerli_mi:
                return format_html('<span style="color:green;">✅ Aktif</span>')
            return format_html('<span style="color:red;">❌ Pasif</span>')
        except Abonelik.DoesNotExist:
            return format_html('<span style="color:orange;">⚠️ Abonelik Yok</span>')
    abonelik_durumu.short_description = "Abonelik"


admin.site.unregister(User)
admin.site.register(User, SovereignUserAdmin)


# ── GunlukAnaliz ──
@admin.register(GunlukAnaliz)
class GunlukAnalizAdmin(admin.ModelAdmin):
    list_display = ('kullanici', 'tarih', 'adet', 'ip_adresi')
    list_filter = ('tarih',)
    readonly_fields = ('tarih',)
    date_hierarchy = 'tarih'

    def kullanici(self, obj):
        return obj.user.username if obj.user else f"Anonim ({obj.ip_adresi})"
    kullanici.short_description = "Kullanıcı"


# ── WatchlistItem ──
@admin.register(WatchlistItem)
class WatchlistItemAdmin(admin.ModelAdmin):
    list_display = ('user', 'sembol', 'pazar', 'eklenme_tarihi')
    list_filter = ('pazar',)
    search_fields = ('user__username', 'sembol')
