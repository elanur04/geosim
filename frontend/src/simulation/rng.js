// Deterministik rastgele sayı üreteci (Mulberry32).
// Tarayıcının Math.random() değeri dışarıdan tohumlanamadığı için deneylerde bu kullanılır.
export const SEEDS = Object.freeze([202601, 202602, 202603, 202604, 202605, 202606, 202607, 202608, 202609, 202610]);

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Aynı tohumdan bağımsız akışlar türetir (varışlar ayrı, kaynak/hedef seçimi ayrı).
// Böylece senaryolar arasında aynı varış zamanları ve aynı kaynak/hedef çiftleri kullanılır.
export function deriveSeed(seed, stream) {
  return (Math.imul(seed >>> 0, 0x9e3779b1) ^ Math.imul(stream + 1, 0x85ebca6b)) >>> 0;
}

// Poisson sürecinde iki varış arasındaki süre (üstel dağılım).
export function exponential(rng, rate) {
  return -Math.log(1 - rng()) / rate;
}
