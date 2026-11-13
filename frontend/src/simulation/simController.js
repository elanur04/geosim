// Worker mesaj mantığı (saf JS; Worker olmadan test edilebilir).
// sim.worker.js yalnızca postMessage / zamanlayıcı bağlantısını yapar.
import { buildRoadGraph } from './roadGraph.js';
import { nodeWeights } from './demand.js';
import { buildScenarios } from './scenarios.js';
import { Simulation, DT, DENSITIES } from './engine.js';
import { SEEDS } from './rng.js';

const MAX_STEPS_PER_TICK = 100; // yavaş kare durumunda bile arayüzü kilitlememek için üst sınır

export function createSimController(post) {
  let scenarios = null, weights = null, sim = null, playing = false, accumulator = 0;
  const cfg = { scenario: 'S0', rate: DENSITIES.medium, seed: SEEDS[0], speed: 1 };

  const rebuild = () => {
    if (!scenarios) return;
    const sc = scenarios[cfg.scenario] ?? scenarios.S0;
    sim = new Simulation(sc.graph, weights, { rate: cfg.rate, seed: cfg.seed });
    accumulator = 0;
    emitFrame();
  };

  const emitFrame = () => {
    if (!sim) return;
    const { count, data } = sim.positions();
    const out = data.slice(0, count * 4);
    let speedSum = 0;
    for (let i = 0; i < count; i++) speedSum += out[i * 4 + 3];
    post({ type: 'frame', t: sim.t, count, data: out, origin: sim.graph.origin,
      live: { vehicles: count, backlog: sim.backlog.length, meanSpeedKmh: count ? (speedSum / count) * 3.6 : 0 } }, [out.buffer]);
  };

  function handle(msg) {
    switch (msg.type) {
      case 'init': {
        playing = false;
        const graph = buildRoadGraph(msg.roads);
        if (graph.nodes.length < 2 || !graph.edges.length) {
          post({ type: 'error', message: 'Sürülebilir yol ağı oluşturulamadı (yol verisi yok veya bağlantısız).' }); return;
        }
        const demand = nodeWeights(graph, msg.buildings, { floorsMode: msg.floorsMode ?? 'median' });
        weights = demand.weights;
        scenarios = buildScenarios(graph, weights, { closeWayId: msg.closeWayId });
        Object.assign(cfg, { scenario: msg.scenario ?? 'S0', rate: msg.rate ?? cfg.rate, seed: msg.seed ?? cfg.seed });
        post({ type: 'ready', origin: graph.origin, graph: graph.report, demand: { ...demand, weights: undefined },
          scenarios: Object.fromEntries(Object.entries(scenarios).map(([k, v]) => [k, { description: v.description, closure: v.closure, connector: v.connector }])) });
        rebuild(); break;
      }
      case 'configure': {
        for (const k of ['scenario', 'rate', 'seed', 'speed']) if (msg[k] !== undefined) cfg[k] = msg[k];
        if (['scenario', 'rate', 'seed'].some((k) => msg[k] !== undefined)) rebuild();
        break;
      }
      case 'play': playing = true; break;
      case 'pause': playing = false; break;
      case 'reset': playing = false; rebuild(); break;
      default: break;
    }
  }

  // elapsedSeconds: gerçek zaman; simülasyon zamanı cfg.speed ile ölçeklenir ve sabit DT adımlarına bölünür.
  function tick(elapsedSeconds) {
    if (!playing || !sim) return false;
    // Birikim adım sayısı cinsinden tutulur: 0,1 s'yi tekrar tekrar çıkarmak kayan nokta hatası biriktirir.
    accumulator += (Math.min(elapsedSeconds, 0.25) * cfg.speed) / DT;
    let steps = Math.min(Math.floor(accumulator + 1e-9), MAX_STEPS_PER_TICK);
    accumulator = steps === MAX_STEPS_PER_TICK ? 0 : accumulator - steps;
    for (let i = 0; i < steps; i++) sim.step();
    if (steps) emitFrame();
    return steps > 0;
  }

  return { handle, tick, get playing() { return playing; } };
}
