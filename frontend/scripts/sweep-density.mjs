// Yoğunluk ön koşusu: serbest akıştan tıkanıklığa geçişin hangi varış hızında olduğunu bulur.
// Araştırma önerisi yoğunluk düzeylerinin ön koşularla sınanıp deneylerden önce dondurulmasını ister.
//   node scripts/sweep-density.mjs --osm osm.geojson [--buildings hybrid.geojson] [--rates 0.15,0.35,0.6,1,1.5,2,3]
import { readFileSync } from 'node:fs';
import { buildRoadGraph } from '../src/simulation/roadGraph.js';
import { nodeWeights } from '../src/simulation/demand.js';
import { Simulation } from '../src/simulation/engine.js';
import { describe } from '../src/simulation/scenarios.js';
import { SEEDS } from '../src/simulation/rng.js';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => (a.startsWith('--') ? [...acc, [a.slice(2), all[i + 1]]] : acc), []));
if (!args.osm) { console.error('Kullanım: --osm osm.geojson [--buildings katman.geojson] [--rates 0.15,0.35,...] [--seeds 5]'); process.exit(1); }
const read = (p) => JSON.parse(readFileSync(p, 'utf8')).features;
const graph = buildRoadGraph(read(args.osm).filter((f) => f.properties?.type === 'road'));
const demand = nodeWeights(graph, read(args.buildings ?? args.osm).filter((f) => f.properties?.type === 'building'));
const rates = (args.rates ?? '0.15,0.35,0.6,1,1.5,2,3,4').split(',').map(Number);
const seeds = SEEDS.slice(0, Number(args.seeds ?? 5));

console.log('Graf:', JSON.stringify(graph.report));
console.log('rate(araç/s)  süre(s)  hız(km/s)  gecikme  durma%  kuyruk  ağda  kilit');
for (const rate of rates) {
  const runs = seeds.map((seed) => new Simulation(graph, demand.weights, { rate, seed }).run());
  const m = (k) => describe(runs.map((r) => r[k])).mean;
  console.log(String(rate).padEnd(12), m('meanTravelTimeS')?.toFixed(1).padStart(7), m('meanSpeedKmh')?.toFixed(1).padStart(9),
    `×${m('meanDelayRatio')?.toFixed(2)}`.padStart(8), (100 * m('stoppedShare')).toFixed(1).padStart(7),
    m('backlogAtEnd').toFixed(1).padStart(7), m('inNetworkAtEnd').toFixed(1).padStart(5), runs.filter((r) => r.gridlock).length);
}
