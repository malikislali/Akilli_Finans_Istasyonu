// sembol_format.js — Ham API sembollerini (GC=F, THYAO.IS gibi) kullanıcı
// dostu görünen isimlere çevirir. SADECE GÖRÜNTÜLEME içindir — API
// çağrılarında, takip listesi eklerken vs. HER ZAMAN ham sembol kullanılmalı.

const EMTIA_ISIM_HARITASI = {
  'GC=F': 'ALTIN',
  'SI=F': 'GÜMÜŞ',
  'CL=F': 'HAM PETROL',
  'BZ=F': 'BRENT PETROL',
  'NG=F': 'DOĞALGAZ',
  'PL=F': 'PLATİN',
  'HG=F': 'BAKIR',
  'PA=F': 'PALADYUM',
};

function sembolGoster(sembol, pazar) {
  if (pazar === 'EMTIA' && EMTIA_ISIM_HARITASI[sembol]) {
    return EMTIA_ISIM_HARITASI[sembol];
  }
  if (pazar === 'TR_HISSE' && sembol.endsWith('.IS')) {
    return sembol.slice(0, -3);
  }
  return sembol;
}