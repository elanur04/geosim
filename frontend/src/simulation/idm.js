// Akıllı Sürücü Modeli (Intelligent Driver Model; Treiber vd., 2000).
// Şehir içi parametreler araştırma önerisinde sabitlenmiştir.
export const IDM_DEFAULTS = Object.freeze({
  v0: 13.9,   // arzulanan hız (m/s)
  T: 1.5,     // güvenli takip süresi (s)
  s0: 2.0,    // durağan minimum mesafe (m)
  a: 1.5,     // maksimum ivme (m/s²)
  b: 2.0,     // konforlu yavaşlama (m/s²)
  delta: 4,   // ivme üssü
  length: 4.5 // araç boyu (m)
});

// gap: öndeki aracın/engelin arka tamponuna net mesafe (m); leaderSpeed: öndekinin hızı (m/s).
// Öndeki yoksa gap = Infinity verin (serbest yol).
export function idmAcceleration(v, vDesired, gap, leaderSpeed, p = IDM_DEFAULTS) {
  const free = 1 - Math.pow(v / vDesired, p.delta);
  if (!Number.isFinite(gap)) return p.a * free;
  const dv = v - leaderSpeed;
  const sStar = p.s0 + Math.max(0, v * p.T + (v * dv) / (2 * Math.sqrt(p.a * p.b)));
  const interaction = Math.pow(sStar / Math.max(gap, 0.01), 2);
  return p.a * (free - interaction);
}
