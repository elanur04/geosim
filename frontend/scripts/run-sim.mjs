// Arşivlenmiş bir analizden (osm.geojson + bina katmanı) çevrimdışı trafik deneyi çalıştırır.
//   node scripts/run-sim.mjs --osm ../backend/data/analysis_runs/<id>/osm.geojson \
//        --buildings ../backend/data/analysis_runs/<id>/hybrid.geojson --out sonuc.json [--seeds 10] [--floors median|one]
import { readFileSync, writeFileSync } from 'node:fs';
import { buildRoadGraph } from '../src/simulation/roadGraph.js';
import { nodeWeights } from '../src/simulation/demand.js';
import { buildScenarios, runBatch } from '../src/simulation/scenarios.js';
import { SEEDS } from '../src/simulation/rng.js';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => (a.startsWith('--') ? [...acc, [a.slice(2), all[i + 1]]] : acc), []));
if (!args.osm) { console.error('Kullanım: --osm osm.geojson [--buildings katman.geojson] [--out sonuc.json] [--seeds 10] [--floors median|one]'); process.exit(1); }

const read = (p) => JSON.parse(readFileSync(p, 'utf8')).features;
const roads = read(args.osm).filter((f) => f.properties?.type === 'road');
const buildings = read(args.buildings ?? args.osm).filter((f) => f.properties?.type === 'building');
const graph = buildRoadGraph(roads);
const main = new Set(graph.nodes.map((n) => n.index));
const demand = nodeWeights(graph, buildings, { floorsMode: args.floors ?? 'median', onlyNodes: main });
const scenarios = buildScenarios(graph, demand.weights);
const seeds = SEEDS.slice(0, Number(args.seeds ?? 10));

console.log('Graf:', JSON.stringify(graph.report));
console.log('Talep:', JSON.stringify({ ...demand, weights: undefined }));
const { runs, cells } = runBatch(scenarios, demand.weights, { seeds });
for (const c of cells) {
  const f = (d, k = 1) => (d.mean === null ? '—' : `${d.mean.toFixed(k)} ± ${d.sd.toFixed(k)}`);
  console.log(`${c.scenario} ${c.density.padEnd(6)} süre ${f(c.meanTravelTimeS)} s | hız ${f(c.meanSpeedKmh)} km/s | gecikme ×${f(c.meanDelayRatio, 2)} | ulaşılamaz ${f(c.unroutable)} | kilit ${c.gridlockRuns}`);
}
if (args.out) {
  writeFileSync(args.out, JSON.stringify({ graph: graph.report, demand: { ...demand, weights: undefined },
    scenarios: Object.fromEntries(Object.entries(scenarios).map(([k, v]) => [k, { description: v.description, closure: v.closure, connector: v.connector }])),
    cells, runs }, null, 2));
}
