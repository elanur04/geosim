import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createSimController } from '../src/simulation/simController.js';
import { SEEDS } from '../src/simulation/rng.js';

const LON0 = 32.34, LAT0 = 41.63;
const DX = 1 / (111320 * Math.cos(LAT0 * Math.PI / 180)), DY = 1 / 111320;
const at = (x, y) => [LON0 + x * DX, LAT0 + y * DY];
const road = (id, pts, nodes) => ({ type: 'Feature', id, geometry: { type: 'LineString', coordinates: pts.map(([x, y]) => at(x, y)) },
  properties: { type: 'road', highway: 'residential', oneway: 'no', osm_id: id, osm_nodes: nodes } });
const roads = () => {
  const out = [];
  for (let i = 0; i < 4; i++) {
    out.push(road(`h${i}`, [0, 1, 2, 3].map((c) => [c * 100, i * 100]), [0, 1, 2, 3].map((c) => i * 10 + c)));
    out.push(road(`v${i}`, [0, 1, 2, 3].map((r) => [i * 100, r * 100]), [0, 1, 2, 3].map((r) => r * 10 + i)));
  }
  return out;
};
const buildings = () => Array.from({ length: 16 }, (_, i) => {
  const [x, y] = [(i % 4) * 100 + 20, Math.floor(i / 4) * 100 + 20];
  return { type: 'Feature', id: `b${i}`, properties: { type: 'building', source: 'osm', geometry_source: 'overpass', floors: 4 },
    geometry: { type: 'Polygon', coordinates: [[at(x - 12, y - 12), at(x + 12, y - 12), at(x + 12, y + 12), at(x - 12, y + 12), at(x - 12, y - 12)]] } };
});

test('Worker denetleyicisi: init -> ready, oynat -> kareler, sıfırla, senaryo ve hata mesajı', () => {
  const sent = [], ctl = createSimController((m) => sent.push(m));
  ctl.handle({ type: 'init', roads: roads(), buildings: buildings(), rate: 0.6, seed: SEEDS[0] });
  assert.equal(sent[0].type, 'ready');
  assert.deepEqual(Object.keys(sent[0].scenarios), ['S0', 'S1', 'S2']);
  assert.equal(sent[0].graph.nodes, 16);
  assert.equal(sent[1].type, 'frame');
  assert.equal(sent[1].t, 0);
  assert.equal(ctl.tick(1), false); // oynatılmıyorsa ilerlemez
  ctl.handle({ type: 'play' });
  for (let i = 0; i < 4; i++) assert.equal(ctl.tick(0.25), true);
  const frame = sent.at(-1);
  assert.ok(Math.abs(frame.t - 1) < 1e-9); // 1 s gerçek zaman = 10 sabit adım
  ctl.handle({ type: 'configure', speed: 1 });
  ctl.tick(60); // uzun duraklama tek karede en fazla 0,25 s ilerletir
  assert.ok(Math.abs(sent.at(-1).t - 1.2) < 1e-9); // 0,25 s sınırı = 2 adım
  ctl.handle({ type: 'reset' }); ctl.handle({ type: 'play' });
  assert.ok(frame.data instanceof Float32Array && frame.data.length === frame.count * 4);
  ctl.handle({ type: 'configure', speed: 5 });
  ctl.tick(0.2); // 0,2 s × 5 = 1 s simülasyon
  assert.ok(Math.abs(sent.at(-1).t - 1) < 1e-9);
  ctl.handle({ type: 'configure', scenario: 'S1' });
  assert.equal(sent.at(-1).t, 0); // senaryo değişince yeniden kurulur
  ctl.handle({ type: 'reset' });
  assert.equal(ctl.playing, false);
  const bad = [];
  createSimController((m) => bad.push(m)).handle({ type: 'init', roads: [], buildings: [] });
  assert.equal(bad[0].type, 'error');
});

test('Worker denetleyicisi aynı tohumda aynı kareleri üretir', () => {
  const run = () => {
    const sent = [], ctl = createSimController((m) => sent.push(m));
    ctl.handle({ type: 'init', roads: roads(), buildings: buildings(), rate: 0.6, seed: SEEDS[4] });
    ctl.handle({ type: 'play' });
    for (let i = 0; i < 30; i++) ctl.tick(0.2);
    return Array.from(sent.at(-1).data);
  };
  const a = run();
  assert.ok(a.length > 0);
  assert.deepEqual(a, run());
});
