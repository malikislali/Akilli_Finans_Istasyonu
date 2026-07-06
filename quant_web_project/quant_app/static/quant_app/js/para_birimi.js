// para_birimi.js — USD/TRY görüntüleme dönüşümü.
// Modlar: 'karma' (varsayılan, her varlık kendi nativ para biriminde),
// 'usd' (her şeyi $ göster), 'try' (her şeyi ₺ göster).

const PARA_BIRIMI_LOCALSTORAGE_ANAHTARI = 'sovereign_para_birimi_modu';
let PARA_BIRIMI_MODU = 'karma';
let USD_TRY_KUR = 35.0;

async function usdTryKuruYukle() {
  try {
    const resp = await fetch('/api/usd-try-kuru/');
    const veri = await resp.json();
    if (veri.basarili) USD_TRY_KUR = veri.kur;
  } catch (err) { console.error('Kur yüklenemedi:', err); }
}

function paraBirimiModunuYukle() {
  try {
    PARA_BIRIMI_MODU = localStorage.getItem(PARA_BIRIMI_LOCALSTORAGE_ANAHTARI) || 'karma';
  } catch (err) { PARA_BIRIMI_MODU = 'karma'; }
}

function paraBirimiModunuKaydet() {
  try { localStorage.setItem(PARA_BIRIMI_LOCALSTORAGE_ANAHTARI, PARA_BIRIMI_MODU); } catch (err) { /* sessiz geç */ }
}

function pazarNativParaBirimi(pazar) {
  return pazar === 'TR_HISSE' ? 'try' : 'usd';
}

function paraBirimiUygula(hamDeger, pazar) {
  const nativ = pazarNativParaBirimi(pazar);
  const hedefBirim = PARA_BIRIMI_MODU === 'karma' ? nativ : PARA_BIRIMI_MODU;
  let deger = hamDeger;
  if (nativ === 'usd' && hedefBirim === 'try') deger = hamDeger * USD_TRY_KUR;
  else if (nativ === 'try' && hedefBirim === 'usd') deger = hamDeger / USD_TRY_KUR;
  return { deger, sembol: hedefBirim === 'try' ? '₺' : '$' };
}

function paraBirimiFormatla(hamDeger, pazar, ondalik = 2) {
  if (hamDeger === null || hamDeger === undefined || isNaN(hamDeger)) return '—';
  const { deger, sembol } = paraBirimiUygula(hamDeger, pazar);
  const yazi = Number(deger).toLocaleString('tr-TR', { minimumFractionDigits: ondalik, maximumFractionDigits: ondalik });
  return sembol === '$' ? `$${yazi}` : `${yazi} ${sembol}`;
}

document.addEventListener('DOMContentLoaded', () => {
  paraBirimiModunuYukle();
  usdTryKuruYukle();
});