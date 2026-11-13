// Deterministik, sabit zaman adımlı (dt = 0,1 s) mikroskobik trafik simülasyonu.
//
// Model özeti
//  - Araçlar Poisson süreciyle gelir; kaynak/hedef düğümleri bina ağırlıklarıyla orantılı seçilir.
//  - Rota ağa girerken serbest akış sürelerine göre Dijkstra ile bir kez hesaplanır.
//  - Her yönlü kenar tek şerit gibi işlenir (şerit sayısı verisi saklanır ama kapasiteye katılmaz).
//  - Araç takibi IDM ile; önde araç yoksa rota boyunca en fazla 150 m / 6 kenar ileriye bakılır.
//  - Kavşak: düğüm başına en az `nodeHeadway` saniye arayla bir geçiş; aynı adımda birden çok araç
//    geliyorsa yol sınıfı yüksek olan, sonra daha uzun bekleyen geçer. Sinyal/dönüş hızı modeli yok.
//  Kaynak/hedef ve varış zamanları iki ayrı Mulberry32 akışından gelir; böylece senaryolar aynı
//  talep dizisiyle karşılaştırılır (ortak rastgele sayılar).
import { IDM_DEFAULTS, idmAcceleration } from './idm.js';
import { mulberry32, deriveSeed, exponential } from './rng.js';
import { makeSampler } from './demand.js';
import { shortestRoute } from './routing.js';
import { makeProjection, pointAlong } from './geo.js';

export const DT = 0.1;
export const DENSITIES = Object.freeze({ low: 0.15, medium: 0.35, high: 0.6 }); // araç/s (başlangıç değerleri)
const LOOKAHEAD_M = 150, LOOKAHEAD_EDGES = 6, ENTRY_MIN_GAP = 1.0, GRIDLOCK_SECONDS = 60;

export class Simulation {
  constructor(graph, weights, options = {}) {
    const { rate = DENSITIES.medium, seed = 202601, warmup = 60, measure = 300, dt = DT, nodeHeadway = 2.0,
      idm = IDM_DEFAULTS } = options;
    this.graph = graph; this.rate = rate; this.seed = seed; this.warmup = warmup; this.measure = measure;
    this.dt = dt; this.nodeHeadway = nodeHeadway; this.idm = { ...IDM_DEFAULTS, ...idm };
    this.sample = makeSampler(weights);
    this.rngArrival = mulberry32(deriveSeed(seed, 0));
    this.rngOd = mulberry32(deriveSeed(seed, 1));
    this.t = 0; this.tick = 0;
    this.nextArrival = exponential(this.rngArrival, rate);
    this.lanes = graph.edges.map(() => []);            // kenar -> araçlar (öndeki araç 0. indekste)
    this.backlog = [];                                  // ağa girmeyi bekleyen araçlar (FIFO)
    this.nodeFreeAt = new Float64Array(graph.nodes.length);
    this.routeCache = new Map();
    this.nextId = 1; this.projection = makeProjection(graph.origin.lon0, graph.origin.lat0);
    this.stats = { arrivals: 0, unroutable: 0, entered: 0, completedAll: 0, overlaps: 0, gridlockSeconds: 0, gridlock: false,
      completed: 0, sumTravel: 0, sumFree: 0, sumDistance: 0, sumQueue: 0, queued: 0,
      speedSum: 0, vehicleTicks: 0, stoppedTicks: 0, delayRatioSum: 0 };
    this.stoppedStreak = 0;
  }

  get vehicleCount() { let n = 0; for (const l of this.lanes) n += l.length; return n; }
  get finished() { return this.t >= this.warmup + this.measure - 1e-9; }

  _route(o, d) {
    const key = o * this.graph.nodes.length + d;
    if (!this.routeCache.has(key)) this.routeCache.set(key, shortestRoute(this.graph, o, d));
    return this.routeCache.get(key);
  }

  _spawn(upTo) {
    while (this.nextArrival < upTo) {
      const origin = this.sample(this.rngOd());
      let dest = this.sample(this.rngOd());
      for (let guard = 0; dest === origin && guard < 50; guard++) dest = this.sample(this.rngOd());
      const created = this.nextArrival;
      this.nextArrival += exponential(this.rngArrival, this.rate);
      this.stats.arrivals++;
      const route = origin === dest ? null : this._route(origin, dest);
      if (!route || !route.edges.length) { this.stats.unroutable++; continue; }
      const distance = route.edges.reduce((s, ei) => s + this.graph.edges[ei].length, 0);
      this.backlog.push({ id: this.nextId++, route: route.edges, routeIdx: 0, pos: 0, v: 0, created, entered: NaN,
        freeFlow: route.time, distance, waitSince: null, edge: route.edges[0] });
    }
  }

  _insertFromBacklog(now) {
    const keep = [];
    for (const veh of this.backlog) {
      const e = this.graph.edges[veh.route[0]], lane = this.lanes[veh.route[0]], rear = lane[lane.length - 1];
      const vEntry = rear ? Math.min(e.speed, rear.v) : e.speed;
      const free = !rear || rear.pos - this.idm.length - 0 >= this.idm.s0 + vEntry * this.idm.T * 0.5 + ENTRY_MIN_GAP;
      if (!free) { keep.push(veh); continue; }
      veh.pos = 0; veh.v = Math.min(vEntry, 5); veh.entered = now; veh.edge = veh.route[0];
      lane.push(veh);
      if (now >= this.warmup && now < this.warmup + this.measure) { this.stats.sumQueue += now - veh.created; this.stats.queued++; }
      this.stats.entered++;
    }
    this.backlog = keep;
  }

  // Öndeki araca (veya kavşak kapısına) net mesafe ve hız
  _leader(veh, k, lane, edge) {
    if (k > 0) { const l = lane[k - 1]; return { gap: l.pos - this.idm.length - veh.pos, speed: l.v }; }
    let dist = edge.length - veh.pos;
    const nextIdx = veh.routeIdx + 1;
    if (nextIdx >= veh.route.length) return { gap: Infinity, speed: 0 }; // hedef kenar: serbest
    // Kavşak kapısı, araç oraya varana kadar da meşgulse durağan engel gibi davranır.
    if (this.nodeFreeAt[edge.to] > this.t + dist / Math.max(veh.v, 0.5)) return { gap: Math.max(dist, 0), speed: 0 };
    for (let j = nextIdx, n = 0; j < veh.route.length && n < LOOKAHEAD_EDGES && dist < LOOKAHEAD_M; j++, n++) {
      const ahead = this.lanes[veh.route[j]];
      if (ahead.length) { const l = ahead[ahead.length - 1]; return { gap: dist + l.pos - this.idm.length, speed: l.v }; }
      dist += this.graph.edges[veh.route[j]].length;
    }
    return { gap: Infinity, speed: 0 };
  }

  step() {
    const dt = this.dt, now = (this.tick + 1) * dt, g = this.graph, p = this.idm; // tick'ten türet: kayan nokta kayması olmasın
    this._spawn(now);
    this._insertFromBacklog(now);

    // 1) İvmeler (eski durumdan, eşzamanlı güncelleme)
    const acc = new Map();
    for (let ei = 0; ei < this.lanes.length; ei++) {
      const lane = this.lanes[ei], edge = g.edges[ei];
      for (let k = 0; k < lane.length; k++) {
        const veh = lane[k], { gap, speed } = this._leader(veh, k, lane, edge);
        acc.set(veh.id, idmAcceleration(veh.v, edge.speed, gap, speed, p));
      }
    }

    // 2) Hız/konum ilerletme ve çakışma koruması
    const crossers = [];
    let speedSum = 0, count = 0, stopped = 0;
    for (let ei = 0; ei < this.lanes.length; ei++) {
      const lane = this.lanes[ei], edge = g.edges[ei];
      for (let k = 0; k < lane.length; k++) {
        const veh = lane[k], vNew = Math.max(0, veh.v + acc.get(veh.id) * dt);
        veh.pos += 0.5 * (veh.v + vNew) * dt; veh.v = vNew;
        if (k > 0) {
          const lead = lane[k - 1], maxPos = lead.pos - p.length;
          if (veh.pos > maxPos) { if (veh.pos - maxPos > 0.5) this.stats.overlaps++; veh.pos = Math.max(0, maxPos); veh.v = Math.min(veh.v, lead.v); }
        }
      }
      if (lane.length && lane[0].pos >= edge.length) crossers.push({ veh: lane[0], ei });
    }

    // 3) Kenar sonuna ulaşanlar: hedefte çık, değilse kavşak kapısından geçiş talep et
    crossers.sort((a, b) => (g.edges[b.ei].rank - g.edges[a.ei].rank)
      || ((a.veh.waitSince ?? now) - (b.veh.waitSince ?? now)) || (a.veh.id - b.veh.id));
    for (const { veh, ei } of crossers) {
      const edge = g.edges[ei], lane = this.lanes[ei];
      if (veh.routeIdx + 1 >= veh.route.length) { // varış
        lane.shift(); this._complete(veh, now); continue;
      }
      const overflow = veh.pos - edge.length, nextEi = veh.route[veh.routeIdx + 1], next = this.lanes[nextEi];
      const rear = next[next.length - 1];
      const open = this.nodeFreeAt[edge.to] <= now && (!rear || rear.pos - p.length - overflow >= ENTRY_MIN_GAP);
      if (open) {
        lane.shift(); veh.routeIdx++; veh.edge = nextEi; veh.pos = Math.min(overflow, g.edges[nextEi].length);
        veh.waitSince = null; next.push(veh); this.nodeFreeAt[edge.to] = now + this.nodeHeadway;
      } else { veh.pos = edge.length; veh.v = 0; if (veh.waitSince === null) veh.waitSince = now; }
    }

    // 4) Ölçüm (yalnızca ölçüm penceresinde) ve kilitlenme tespiti
    const inWindow = now > this.warmup && now <= this.warmup + this.measure + 1e-9;
    for (const lane of this.lanes) for (const veh of lane) { count++; speedSum += veh.v; if (veh.v < 0.1) stopped++; }
    if (inWindow) { this.stats.speedSum += speedSum; this.stats.vehicleTicks += count; this.stats.stoppedTicks += stopped; }
    if (count > 0 && speedSum < 1e-6) this.stoppedStreak += dt; else this.stoppedStreak = 0;
    if (this.stoppedStreak >= GRIDLOCK_SECONDS) this.stats.gridlock = true;
    this.t = now; this.tick++;
  }

  _complete(veh, now) {
    this.stats.completedAll++;
    if (now > this.warmup && now <= this.warmup + this.measure + 1e-9) {
      const travel = now - veh.entered, s = this.stats;
      s.completed++; s.sumTravel += travel; s.sumFree += veh.freeFlow; s.sumDistance += veh.distance;
      s.delayRatioSum += travel / Math.max(veh.freeFlow, 1e-9);
    }
  }

  run() { while (!this.finished) this.step(); return this.summary(); }

  summary() {
    const s = this.stats;
    return {
      seed: this.seed, rate: this.rate, warmup: this.warmup, measure: this.measure,
      arrivals: s.arrivals, unroutable: s.unroutable, entered: s.entered,
      completedInWindow: s.completed, completedTotal: s.completedAll,
      meanTravelTimeS: s.completed ? s.sumTravel / s.completed : null,
      meanFreeFlowTimeS: s.completed ? s.sumFree / s.completed : null,
      meanDelayRatio: s.completed ? s.delayRatioSum / s.completed : null,
      meanSpeedMs: s.vehicleTicks ? s.speedSum / s.vehicleTicks : null,
      meanSpeedKmh: s.vehicleTicks ? (s.speedSum / s.vehicleTicks) * 3.6 : null,
      tripMeanSpeedMs: s.sumTravel ? s.sumDistance / s.sumTravel : null,
      stoppedShare: s.vehicleTicks ? s.stoppedTicks / s.vehicleTicks : null,
      meanQueueDelayS: s.queued ? s.sumQueue / s.queued : 0,
      backlogAtEnd: this.backlog.length, inNetworkAtEnd: this.vehicleCount,
      overlapEvents: s.overlaps, gridlock: s.gridlock,
    };
  }

  // Çizim için araç durumu: [x, y, yön(rad), hız] — x,y graf orijinine göre metre ofseti.
  positions(target) {
    const n = this.vehicleCount, data = target && target.length >= n * 4 ? target : new Float32Array(n * 4);
    let i = 0;
    for (let ei = 0; ei < this.lanes.length; ei++) {
      const e = this.graph.edges[ei];
      for (const veh of this.lanes[ei]) {
        const pt = pointAlong(e.pts, e.cum, veh.pos);
        data[i++] = pt.x; data[i++] = pt.y; data[i++] = pt.heading; data[i++] = veh.v;
      }
    }
    return { count: n, data };
  }
}

export function runSimulation(graph, weights, options) {
  return new Simulation(graph, weights, options).run();
}
