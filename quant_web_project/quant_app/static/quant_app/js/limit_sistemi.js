// limit_sistemi.js — Ücretsiz üyelik kısıtlamaları için ortak modal + toast.
// Hem market.html hem dashboard.html tarafından dahil edilir.

function limitModalAc(mesaj, linkUrl) {
  const mesajEl = document.getElementById('limit-modal-mesaj');
  const linkEl = document.getElementById('limit-modal-link');
  if (!mesajEl || !linkEl) return;
  mesajEl.textContent = mesaj;
  linkEl.href = linkUrl;
  document.getElementById('limit-modal-overlay').classList.remove('hidden');
}

function limitModalKapat() {
  const overlay = document.getElementById('limit-modal-overlay');
  if (overlay) overlay.classList.add('hidden');
}

let _ucretliToastTimeout = null;
function ucretliToastGoster() {
  const toast = document.getElementById('ucretli-toast');
  if (!toast) return;
  toast.classList.remove('hidden');
  void toast.offsetWidth; // reflow — transition'ın tetiklenmesi için
  toast.classList.add('goster');
  clearTimeout(_ucretliToastTimeout);
  _ucretliToastTimeout = setTimeout(() => {
    toast.classList.remove('goster');
    setTimeout(() => toast.classList.add('hidden'), 250);
  }, 2000);
}

document.addEventListener('DOMContentLoaded', () => {
  const overlay = document.getElementById('limit-modal-overlay');
  if (overlay) {
    overlay.addEventListener('click', (e) => {
      if (e.target.id === 'limit-modal-overlay') limitModalKapat();
    });
  }
});

// Kalan hak yazısını formatlar. limitBilgisi = {kullanilan, limit, kalan}
function hakEtiketiHtmlUret(emoji, etiket, limitBilgisi) {
  const tukendi = limitBilgisi.kalan <= 0;
  return `<span class="hak-bilgisi-etiketi ${tukendi ? 'tukendi' : ''}">${emoji} ${etiket}: ${limitBilgisi.kalan}/${limitBilgisi.limit} hakkınız kaldı</span>`;
}

// ---------------- OTURUM SÜRESİ DOLMA KONTROLÜ ----------------
// Herhangi bir fetch() isteği, oturum süresi dolduğu için login
// sayfasına yönlendirilmiş HTML dönerse (AJAX bunu görünmez şekilde
// alır), kullanıcıyı zorla login'e gönder.
const _orijinalFetch = window.fetch;
window.fetch = async function (...args) {
  const response = await _orijinalFetch.apply(this, args);
  const istekUrl = typeof args[0] === 'string' ? args[0] : (args[0]?.url || '');
  // Sadece kendi API isteklerimizi kontrol et (dış servisleri değil)
  if (istekUrl.startsWith('/') && response.redirected && response.url.includes('/login/')) {
    window.location.href = '/login/';
  }
  return response;
};