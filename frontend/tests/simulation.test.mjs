import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mulberry32, deriveSeed, exponential, SEEDS } from '../src/simulation/rng.js';
import { buildRoadGraph, closeEdges, addConnector } from '../src/simulation/roadGraph.js';
import { shortestRoute } from '../src/simulation/routing.js';
import { idmAcceleration, IDM_DEFAULTS } from '../src/simulation/idm.js';
import { nodeWeights, makeSampler } from '../src/simulation/demand.js';
import { Simulation, DT } from '../src/simulation/engine.js';
import { buildScenarios, describe as stats } from '../src/simulation/scenarios.js';

const LON0 = 32.34, LAT0 = 41.63;
const DX = 1 / (111320 * Math.cos(LAT0 * Math.PI / 180)), DY = 1 / 111320;
const at = (xm, ym) => [LON0 + xm * DX, LAT0 + ym * DY];
const road = (id, pts, extra = {}, nodes = null) => ({ type: 'Feature', id, geometry: { type: 'LineString', coordinates: pts.map(([x, y]) => at(x, y)) },
  properties: { type: 'road', highway: 'residential', oneway: 'no', osm_id: id, ...(nodes ? { osm_nodes: nodes } : {}), ...extra } });
const square = (x, y, s = 10, props = {}) => ({ type: 'Feature', id: `b${x}_${y}`, properties: { type: 'building', source: 'osm', geometry_source: 'overpass', ...props },
  geometry: { type: 'Polygon', coordinates: [[at(x - s, y - s), at(x + s, y - s), at(x + s, y + s), at(x - s, y + s), at(x - s, y - s)]] } });

// 4x4 ızgara, 100 m aralık, OSM düğüm kimlikli çift yönlü yollar
function gridRoads(n = 4, step = 100) {
  const roads = [];
  for (let r = 0; r < n; r++) roads.push(road(`h${r}`, Array.from({ length: n }, (_, c) => [c * step, r * step]), {}, Array.from({ length: n }, (_, c) => r * 10 + c)));
  for (let c = 0; c < n; c++) roads.push(road(`v${c}`, Array.from({ length: n }, (_, r) => [c * step, r * step]), {}, Array.from({ length: n }, (_, r) => r * 10 + c)));
  return roads;
}
const gridBuildings = (n = 4, step = 100) => Array.from({ length: n * n }, (_, i) => square((i % n) * step + 20, Math.floor(i / n) * step + 20, 12, { floors: 4 }));

test('Mulberry32 aynı tohumda aynı diziyi üretir, akışlar bağımsızdır', () => {
  const a = mulberry32(SEEDS[0]), b = mulberry32(SEEDS[0]);
  for (let i = 0; i < 50; i++) { const x = a(); assert.equal(x, b()); assert.ok(x >= 0 && x < 1); }
  assert.notEqual(deriveSeed(SEEDS[0], 0), deriveSeed(SEEDS[0], 1));
  assert.notEqual(mulberry32(SEEDS[0])(), mulberry32(SEEDS[1])());
  const r = mulberry32(1); let sum = 0; for (let i = 0; i < 20000; i++) sum += exponential(r, 0.5);
  assert.ok(Math.abs(sum / 20000 - 2) < 0.1); // ortalama 1/λ
});

test('ızgara grafı: 16 düğüm, 24 çift yönlü bağlantı = 48 yönlü kenar', () => {
  const g = buildRoadGraph(gridRoads());
  assert.equal(g.nodes.length, 16);
  assert.equal(g.edges.length, 48);
  assert.equal(g.report.largestNodeShare, 1);
  for (const e of g.edges) assert.ok(Math.abs(e.length - 100) < 0.5);
});

test('tek yön, yaya yolu ve özel erişim kuralları', () => {
  const roads = [
    road('a', [[0, 0], [100, 0]], { oneway: 'yes' }, [1, 2]),
    road('b', [[100, 0], [100, 100]], {}, [2, 3]),
    road('c', [[100, 100], [0, 0]], {}, [3, 1]),
    road('walk', [[0, 0], [0, 50]], { highway: 'footway' }, [1, 9]),
    road('priv', [[100, 100], [200, 100]], { access: 'private' }, [3, 8]),
    road('rev', [[0, 200], [100, 200]], { oneway: '-1' }, [20, 21]),
  ];
  const g = buildRoadGraph(roads, { pruneToMainComponent: false });
  const has = (from, to) => g.edges.some((e) => g.nodes[e.from].id === `n${from}` && g.nodes[e.to].id === `n${to}`);
  assert.ok(has(1, 2) && !has(2, 1));
  assert.ok(has(21, 20) && !has(20, 21));
  assert.ok(!g.nodes.some((n) => ['n9', 'n8'].includes(n.id)));
});

test('köprü ile alt geçit ortak OSM düğümü yoksa bağlanmaz', () => {
  const roads = [
    road('flat', [[0, 100], [200, 100]], {}, [1, 2]),
    road('bridge', [[100, 0], [100, 200]], { bridge: 'yes', layer: '1' }, [3, 4]),
  ];
  const g = buildRoadGraph(roads, { pruneToMainComponent: false });
  assert.equal(g.nodes.length, 4);
  assert.equal(g.report.components, 2); // iki ayrı çift yönlü yol, birbirine bağlı değil
});

test('kimliksiz yollar: geometrik kesişim ve T-bağlantı düğüm üretir, köprü üretmez', () => {
  const cross = buildRoadGraph([road('x', [[0, 0], [200, 0]]), road('y', [[100, -100], [100, 100]])], { pruneToMainComponent: false });
  assert.equal(cross.nodes.length, 5); // 4 uç + 1 kesişim
  assert.equal(cross.edges.length, 8);
  const tee = buildRoadGraph([road('x', [[0, 0], [200, 0]]), road('stem', [[100, 1], [100, 120]])], { pruneToMainComponent: false });
  assert.equal(tee.nodes.length, 4); // x'in iki ucu, T noktası ve sapağın serbest ucu
  const bridge = buildRoadGraph([road('x', [[0, 0], [200, 0]]), road('y', [[100, -100], [100, 100]], { bridge: 'yes', layer: '1' })], { pruneToMainComponent: false });
  assert.equal(bridge.nodes.length, 4);
});

test('en büyük güçlü bileşene budama: tek yönlü çıkmaz uç atılır', () => {
  const roads = [...gridRoads(3).filter((r) => r.id.startsWith('h') || r.id.startsWith('v')),
    road('sink', [[200, 200], [400, 200]], { oneway: 'yes' }, [22, 99])];
  const g = buildRoadGraph(roads);
  assert.equal(g.nodes.length, 9);
  assert.ok(g.report.droppedEdges >= 1);
  assert.ok(g.report.components > 1);
});

test('Dijkstra serbest akış süresi en kısa rotayı seçer', () => {
  const g = buildRoadGraph(gridRoads());
  const idx = (id) => g.nodes.findIndex((n) => n.id === `n${id}`);
  const r = shortestRoute(g, idx(0), idx(33));
  assert.equal(r.edges.length, 6);
  assert.ok(Math.abs(r.time - 600 / 13.9) < 0.5);
  assert.equal(shortestRoute(g, idx(0), idx(0)).edges.length, 0);
});

test('yol kapatma ve yeni bağlantı senaryo işlemleri', () => {
  const g = buildRoadGraph(gridRoads());
  const closed = closeEdges(g, new Set(['h0']));
  assert.equal(closed.edges.length, g.edges.length - 6);
  assert.equal(g.edges.length, 48); // özgün graf değişmez
  const idx = (id) => g.nodes.findIndex((n) => n.id === `n${id}`);
  const a = g.nodes[idx(0)], b = g.nodes[idx(33)];
  const linked = addConnector(g, [a.lon, a.lat], [b.lon, b.lat], { maxSnapM: 5 });
  assert.equal(linked.edges.length, 50);
  const r = shortestRoute(linked, idx(0), idx(33));
  assert.equal(r.edges.length, 1);
  assert.throws(() => addConnector(g, at(5000, 5000), at(0, 0)));
});

test('IDM: serbest yolda ivme a, durağan engel önünde fren, denge takip mesafesi', () => {
  assert.ok(Math.abs(idmAcceleration(0, 13.9, Infinity, 0) - IDM_DEFAULTS.a) < 1e-9);
  assert.ok(Math.abs(idmAcceleration(13.9, 13.9, Infinity, 0)) < 1e-9);
  assert.ok(idmAcceleration(13.9, 13.9, 20, 0) < -IDM_DEFAULTS.b);
  const sEq = (IDM_DEFAULTS.s0 + 10 * IDM_DEFAULTS.T) / Math.sqrt(1 - Math.pow(10 / 13.9, 4));
  assert.ok(Math.abs(idmAcceleration(10, 13.9, sEq, 10)) < 1e-6);
  // durağan engele yaklaşan araç çarpmadan durur
  let v = 13.9, x = 0; const wall = 150;
  for (let i = 0; i < 2000; i++) { const a = idmAcceleration(v, 13.9, wall - x, 0); v = Math.max(0, v + a * DT); x += v * DT; assert.ok(x <= wall); }
  assert.ok(v < 0.05 && wall - x >= IDM_DEFAULTS.s0 - 0.5);
});

test('talep ağırlığı: w_i = w_0 + Σ alan×kat, kat medyanı ve kat=1 duyarlılığı', () => {
  const g = buildRoadGraph(gridRoads());
  const b = [square(10, 10, 10, { floors: 4 }), square(190, 190, 10, { floors: 6 }), square(95, 5, 10), square(5000, 5000, 10, { floors: 4 })];
  const d = nodeWeights(g, b, { baseWeight: 1 });
  assert.equal(d.medianFloors, 4); // kat verisi olan binalar: 4, 6 ve uzaktaki 4 -> medyan 4
  assert.equal(d.buildings, 4);
  assert.equal(d.buildingsWithinRadius, 3); // 5 km uzaktaki bina hiçbir düğüme 75 m yakın değil
  const node0 = g.nodes.findIndex((n) => n.id === 'n0');
  assert.ok(d.weights[node0] > 1 + 400 * 4 * 0.99); // 20x20 m bina × 4 kat
  const one = nodeWeights(g, b, { floorsMode: 'one' });
  assert.equal(one.medianFloors, 1);
  assert.ok(one.weights[node0] < d.weights[node0]);
  const sample = makeSampler(d.weights);
  const counts = new Map(); const r = mulberry32(5);
  for (let i = 0; i < 20000; i++) { const k = sample(r()); counts.set(k, (counts.get(k) || 0) + 1); }
  const total = d.weights.reduce((s, w) => s + w, 0);
  assert.ok(Math.abs(counts.get(node0) / 20000 - d.weights[node0] / total) < 0.02);
});

function setup() {
  const g = buildRoadGraph(gridRoads());
  return { g, demand: nodeWeights(g, gridBuildings()) };
}

test('simülasyon deterministik: aynı tohum aynı sonuç, farklı tohum farklı sonuç', () => {
  const { g, demand } = setup();
  const run = (seed) => new Simulation(g, demand.weights, { rate: 0.35, seed, warmup: 20, measure: 60 }).run();
  assert.deepEqual(run(SEEDS[0]), run(SEEDS[0]));
  assert.notDeepEqual(run(SEEDS[0]), run(SEEDS[1]));
});

test('seyrek trafikte gecikme oranı ~1 ve araçlar çakışmaz', () => {
  const { g, demand } = setup();
  const s = new Simulation(g, demand.weights, { rate: 0.05, seed: SEEDS[2], warmup: 30, measure: 300 }).run();
  assert.ok(s.completedInWindow > 5, `tamamlanan ${s.completedInWindow}`);
  assert.ok(s.meanDelayRatio >= 0.99 && s.meanDelayRatio < 1.5, `gecikme oranı ${s.meanDelayRatio}`);
  assert.equal(s.overlapEvents, 0);
  assert.equal(s.gridlock, false);
});

test('yoğunluk arttıkça hız düşer, süre ve gecikme artar (kilitlenme yok)', () => {
  const { g, demand } = setup();
  const run = (rate) => new Simulation(g, demand.weights, { rate, seed: SEEDS[3], warmup: 60, measure: 300 }).run();
  const low = run(0.15), high = run(0.6);
  assert.ok(high.meanSpeedMs <= low.meanSpeedMs + 1e-9);
  assert.ok(high.meanDelayRatio >= low.meanDelayRatio);
  assert.equal(low.overlapEvents + high.overlapEvents, 0);
  assert.equal(high.gridlock, false);
});

test('senaryolar aynı varış dizisini kullanır (ortak rastgele sayılar)', () => {
  const { g, demand } = setup();
  const sc = buildScenarios(g, demand.weights);
  assert.deepEqual(Object.keys(sc), ['S0', 'S1', 'S2']);
  const arrivals = Object.values(sc).map((s) => new Simulation(s.graph, demand.weights, { rate: 0.35, seed: SEEDS[0], warmup: 20, measure: 100 }).run().arrivals);
  assert.equal(new Set(arrivals).size, 1);
});

test('positions: Float32Array, araç başına 4 değer', () => {
  const { g, demand } = setup();
  const sim = new Simulation(g, demand.weights, { rate: 0.6, seed: SEEDS[0] });
  for (let i = 0; i < 600; i++) sim.step();
  const { count, data } = sim.positions();
  assert.ok(count > 0 && data instanceof Float32Array && data.length >= count * 4);
  for (let i = 0; i < count * 4; i++) assert.ok(Number.isFinite(data[i]));
});

test('istatistik yardımcısı: ortalama, sd ve %95 güven aralığı', () => {
  const s = stats([1, 2, 3, 4, 5]);
  assert.equal(s.mean, 3);
  assert.ok(Math.abs(s.sd - Math.sqrt(2.5)) < 1e-9);
  assert.ok(s.ci95[0] < 3 && s.ci95[1] > 3);
});
