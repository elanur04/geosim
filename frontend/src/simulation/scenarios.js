// Senaryo üretimi (S0 mevcut ağ, S1 yol kapatma, S2 yeni bağlantı) ve toplu deney çalıştırma.
//
// S1 ve S2 için seçim kuralları deterministiktir.
// Elle seçim için closeEdges / addConnector doğrudan kullanılabilir.
import { Simulation, DENSITIES } from './engine.js';
import { SEEDS, mulberry32, deriveSeed } from './rng.js';
import { makeSampler } from './demand.js';
import { shortestRoute } from './routing.js';
import { closeEdges, addConnector, connectivity } from './roadGraph.js';

// S0'daki talep örneğinde en çok kullanılan yol (wayId) — S1 için kapatma adayı.
export function proposeClosure(graph, weights, { samples = 400, seed = SEEDS[0] } = {}) {
  const sample = makeSampler(weights), rng = mulberry32(deriveSeed(seed, 7)), use = new Map();
  for (let i = 0; i < samples; i++) {
    const o = sample(rng()), d = sample(rng());
    if (o === d) continue;
    const r = shortestRoute(graph, o, d);
    if (r) for (const ei of r.edges) { const w = graph.edges[ei].wayId; use.set(w, (use.get(w) || 0) + 1); }
  }
  const ranked = [...use.entries()].sort((a, b) => b[1] - a[1] || String(a[0]).localeCompare(String(b[0])));
  // Ağı bölmeyen en çok kullanılan yol seçilir; böylece tüm kaynak/hedef çiftleri ulaşılabilir kalır
  // ve S0 ile S1 aynı yolculuk kümesi üzerinden karşılaştırılır.
  for (const [wayId, count] of ranked) {
    if (connectivity(closeEdges(graph, new Set([wayId]))).connected) return { wayId, routeShare: count / samples, disconnects: false };
  }
  return ranked.length ? { wayId: ranked[0][0], routeShare: ranked[0][1] / samples, disconnects: true } : null;
}

// S2 için: kuş uçuşu 80–400 m olan, ağ üzerinden en dolambaçlı (ağırlıklı) düğüm çifti.
export function proposeConnector(graph, weights, { minM = 80, maxM = 400, pairs = 600, seed = SEEDS[0] } = {}) {
  const sample = makeSampler(weights), rng = mulberry32(deriveSeed(seed, 11));
  let best = null;
  for (let i = 0; i < pairs; i++) {
    const a = sample(rng()), b = sample(rng());
    if (a === b) continue;
    const euclid = Math.hypot(graph.nodes[a].x - graph.nodes[b].x, graph.nodes[a].y - graph.nodes[b].y);
    if (euclid < minM || euclid > maxM) continue;
    const r = shortestRoute(graph, a, b);
    if (!r) continue;
    const networkM = r.edges.reduce((s, ei) => s + graph.edges[ei].length, 0);
    const score = (networkM / euclid) * Math.sqrt(weights[a] * weights[b]);
    if (!best || score > best.score) best = { from: a, to: b, euclidM: euclid, networkM, detour: networkM / euclid, score };
  }
  return best && { ...best,
    fromLonLat: [graph.nodes[best.from].lon, graph.nodes[best.from].lat],
    toLonLat: [graph.nodes[best.to].lon, graph.nodes[best.to].lat] };
}

export function buildScenarios(graph, weights, { closeWayId, connector } = {}) {
  const scenarios = { S0: { graph, description: 'Mevcut ağ' } };
  const closure = closeWayId ? { wayId: closeWayId } : proposeClosure(graph, weights);
  if (closure) scenarios.S1 = { graph: closeEdges(graph, new Set([closure.wayId])), description: `Yol kapatma (${closure.wayId})`, closure };
  const link = connector ?? proposeConnector(graph, weights);
  if (link) {
    try { scenarios.S2 = { graph: addConnector(graph, link.fromLonLat, link.toLonLat), description: 'Yeni bağlantı yolu', connector: link }; }
    catch { /* uygun bağlantı bulunamadı */ }
  }
  return scenarios;
}

const T_975 = { 1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228 };

export function describe(values) {
  const v = values.filter((x) => Number.isFinite(x)), n = v.length;
  if (!n) return { n: 0, mean: null, sd: null, ci95: null };
  const mean = v.reduce((a, b) => a + b, 0) / n;
  const sd = n > 1 ? Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1)) : 0;
  const half = n > 1 ? (T_975[n - 1] ?? 1.96) * sd / Math.sqrt(n) : null;
  return { n, mean, sd, ci95: half === null ? null : [mean - half, mean + half] };
}

// Senaryo × yoğunluk × tohum koşuları; her hücre için ortalama ± sd ve %95 güven aralığı döner.
export function runBatch(scenarios, weights, options = {}) {
  const { densities = DENSITIES, seeds = SEEDS, warmup = 60, measure = 300, onRun = null } = options;
  const runs = [], cells = [];
  for (const [name, sc] of Object.entries(scenarios)) {
    for (const [level, rate] of Object.entries(densities)) {
      const cell = [];
      for (const seed of seeds) {
        const summary = new Simulation(sc.graph, weights, { rate, seed, warmup, measure }).run();
        const run = { scenario: name, density: level, ...summary };
        runs.push(run); cell.push(run); onRun?.(run);
      }
      cells.push({ scenario: name, density: level, rate, runs: cell.length,
        meanTravelTimeS: describe(cell.map((r) => r.meanTravelTimeS)),
        meanSpeedKmh: describe(cell.map((r) => r.meanSpeedKmh)),
        meanDelayRatio: describe(cell.map((r) => r.meanDelayRatio)),
        stoppedShare: describe(cell.map((r) => r.stoppedShare)),
        unroutable: describe(cell.map((r) => r.unroutable)),
        gridlockRuns: cell.filter((r) => r.gridlock).length });
    }
  }
  return { runs, cells };
}
