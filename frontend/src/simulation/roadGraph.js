// GeoJSON yol çizgilerinden yönlü topolojik yol grafı (düğüm/kenar) üretir.
//
// Bağlantı kuralları:
//  - OSM düğüm kimliği (osm_nodes) olan yollar yalnız ortak OSM düğümünden bağlanır.
//    Düğümü paylaşmayan kesişimler (köprü/alt geçit) bağlanmaz.
//  - Kimliği olmayan yollar (çizilmiş/MVT) için kesişim ve T-bağlantıları geometrik olarak
//    bulunur; köprü, tünel ve farklı layer değerleri birbirine bağlanmaz.
//  - Graf düğümleri kavşaklar ve çıkmaz uçlardır; iki kavşak arası yol bir kenardır.
import { makeProjection, polylineLength } from './geo.js';

export const DEFAULT_SPEED_MS = 13.9; // 50 km/sa (araştırma önerisindeki v0)
export const NON_DRIVABLE = new Set([
  'footway', 'path', 'pedestrian', 'steps', 'cycleway', 'bridleway', 'corridor', 'track',
  'construction', 'proposed', 'raceway', 'bus_guideway', 'elevator', 'platform', 'escape',
]);
// Kavşakta öncelik sırası (büyük sayı = öncelikli).
const CLASS_RANK = {
  motorway: 7, motorway_link: 6, trunk: 6, trunk_link: 5, primary: 5, primary_link: 4,
  secondary: 4, secondary_link: 3, tertiary: 3, tertiary_link: 2, unclassified: 2,
  residential: 1, living_street: 1, service: 0,
};

export function parseMaxspeed(value) {
  if (value === null || value === undefined) return null;
  const m = String(value).trim().match(/^(\d+(?:\.\d+)?)\s*(mph)?$/i);
  if (!m) return null;
  const kmh = parseFloat(m[1]) * (m[2] ? 1.609344 : 1);
  return kmh > 0 ? kmh / 3.6 : null;
}

// 1: çizim yönünde tek yön, -1: ters yönde tek yön, 0: çift yön
export function onewayDirection(props) {
  const o = String(props.oneway ?? 'no').toLowerCase();
  if (o === '-1' || o === 'reverse') return -1;
  if (o === 'yes' || o === 'true' || o === '1') return 1;
  if (props.junction === 'roundabout') return 1;
  if (props.highway === 'motorway' || props.highway === 'motorway_link') return 1;
  return 0;
}

export function isDrivable(props) {
  const hw = props.highway ?? props.class;
  if (!hw || NON_DRIVABLE.has(hw)) return false;
  return !['no', 'private'].includes(String(props.access ?? '').toLowerCase());
}

function lanesOf(props) {
  const n = parseInt(props.lanes, 10);
  return Number.isFinite(n) && n > 0 ? n : 1;
}

const levelKey = (p) => `${p.layer ?? '0'}|${p.bridge ?? 'no'}|${p.tunnel ?? 'no'}`;
const coordKey = (c) => `c${Math.round(c[0] * 1e7)},${Math.round(c[1] * 1e7)}`;

class UnionFind {
  constructor() { this.parent = new Map(); }
  find(k) {
    if (!this.parent.has(k)) this.parent.set(k, k);
    let root = k;
    while (this.parent.get(root) !== root) root = this.parent.get(root);
    while (this.parent.get(k) !== root) { const next = this.parent.get(k); this.parent.set(k, root); k = next; }
    return root;
  }
  union(a, b) { const ra = this.find(a), rb = this.find(b); if (ra !== rb) this.parent.set(rb, ra); }
}

function segmentIntersection(a, b, c, d) {
  const rx = b[0] - a[0], ry = b[1] - a[1], sx = d[0] - c[0], sy = d[1] - c[1];
  const den = rx * sy - ry * sx;
  if (Math.abs(den) < 1e-9) return null;
  const t = ((c[0] - a[0]) * sy - (c[1] - a[1]) * sx) / den;
  const u = ((c[0] - a[0]) * ry - (c[1] - a[1]) * rx) / den;
  return t >= 0 && t <= 1 && u >= 0 && u <= 1 ? { t, u } : null;
}

function projectOnSegment(p, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1], len2 = dx * dx + dy * dy;
  const t = len2 === 0 ? 0 : Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2));
  const x = a[0] + t * dx, y = a[1] + t * dy;
  return { t, x, y, distance: Math.hypot(p[0] - x, p[1] - y) };
}

// Kimliği olmayan yollar için kesişim / T-bağlantı düğümlerini ekler (union-find ile birleştirir).
function connectGeometrically(ways, uf, toleranceM) {
  const inserts = ways.map(() => []);
  let fresh = 0;
  const cell = 60, grid = new Map();
  const segs = [];
  ways.forEach((w, wi) => {
    for (let s = 0; s + 1 < w.pts.length; s++) {
      const id = segs.length; segs.push({ wi, s });
      const [a, b] = [w.pts[s], w.pts[s + 1]];
      const x0 = Math.floor((Math.min(a[0], b[0]) - toleranceM) / cell), x1 = Math.floor((Math.max(a[0], b[0]) + toleranceM) / cell);
      const y0 = Math.floor((Math.min(a[1], b[1]) - toleranceM) / cell), y1 = Math.floor((Math.max(a[1], b[1]) + toleranceM) / cell);
      for (let gx = x0; gx <= x1; gx++) for (let gy = y0; gy <= y1; gy++) {
        const k = `${gx},${gy}`; if (!grid.has(k)) grid.set(k, []); grid.get(k).push(id);
      }
    }
  });
  const seen = new Set();
  // (way, segment, t) noktasında düğüm anahtarı döndürür: köşeye yakınsa köşe, değilse yeni düğüm eklenir.
  const keyAt = (wi, s, t, x, y) => {
    const w = ways[wi], len = Math.hypot(w.pts[s + 1][0] - w.pts[s][0], w.pts[s + 1][1] - w.pts[s][1]);
    if (t * len <= 0.05) return w.keys[s];
    if ((1 - t) * len <= 0.05) return w.keys[s + 1];
    const key = `x${fresh++}`;
    inserts[wi].push({ s, t, x, y, key });
    return key;
  };
  for (const ids of grid.values()) {
    for (let i = 0; i < ids.length; i++) for (let j = i + 1; j < ids.length; j++) {
      const A = segs[ids[i]], B = segs[ids[j]];
      if (A.wi === B.wi) continue;
      const wa = ways[A.wi], wb = ways[B.wi];
      if (wa.hasIds && wb.hasIds) continue;
      if (levelKey(wa.props) !== levelKey(wb.props)) continue;
      const pair = `${Math.min(ids[i], ids[j])}-${Math.max(ids[i], ids[j])}`;
      if (seen.has(pair)) continue; seen.add(pair);
      const a = wa.pts[A.s], b = wa.pts[A.s + 1], c = wb.pts[B.s], d = wb.pts[B.s + 1];
      const hit = segmentIntersection(a, b, c, d);
      if (hit) {
        const x = a[0] + (b[0] - a[0]) * hit.t, y = a[1] + (b[1] - a[1]) * hit.t;
        uf.union(keyAt(A.wi, A.s, hit.t, x, y), keyAt(B.wi, B.s, hit.u, x, y));
        continue;
      }
      // T-bağlantı: bir yolun serbest ucu diğerinin segmentine toleransın içinde değiyorsa
      for (const [w, s, other, os, oa, ob] of [[wa, A.s, B.wi, B.s, c, d], [wb, B.s, A.wi, A.s, a, b]]) {
        const endpoints = [];
        if (s === 0) endpoints.push(0);
        if (s === w.pts.length - 2) endpoints.push(w.pts.length - 1);
        for (const vi of endpoints) {
          const pr = projectOnSegment(w.pts[vi], oa, ob);
          if (pr.distance <= toleranceM) uf.union(w.keys[vi], keyAt(other, os, pr.t, pr.x, pr.y));
        }
      }
    }
  }
  // Eklemeleri uygula (segment içinde t'ye göre sıralı)
  ways.forEach((w, wi) => {
    if (!inserts[wi].length) return;
    const bySeg = new Map();
    for (const ins of inserts[wi]) { if (!bySeg.has(ins.s)) bySeg.set(ins.s, []); bySeg.get(ins.s).push(ins); }
    const pts = [], keys = [];
    for (let s = 0; s < w.pts.length; s++) {
      pts.push(w.pts[s]); keys.push(w.keys[s]);
      const list = (bySeg.get(s) || []).sort((p, q) => p.t - q.t);
      let last = null;
      for (const ins of list) {
        if (last && Math.hypot(ins.x - last.x, ins.y - last.y) < 0.05) { uf.union(last.key, ins.key); continue; }
        pts.push([ins.x, ins.y]); keys.push(ins.key); last = ins;
      }
    }
    w.pts = pts; w.keys = keys;
  });
}

function cumulativeLengths(pts) {
  const cum = [0];
  for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
  return cum;
}

export function finalizeGraph(origin, nodes, edges, report = null) {
  const out = nodes.map(() => []), inc = nodes.map(() => []);
  edges.forEach((e, i) => { e.index = i; out[e.from].push(i); inc[e.to].push(i); });
  return { origin, nodes, edges, out, inc, report };
}

// Tarjan (yinelemeli): en büyük güçlü bağlı bileşen
function stronglyConnected(nodeCount, out, edges) {
  const index = new Int32Array(nodeCount).fill(-1), low = new Int32Array(nodeCount), onStack = new Uint8Array(nodeCount);
  const comp = new Int32Array(nodeCount).fill(-1), stack = [];
  let counter = 0, comps = 0;
  for (let root = 0; root < nodeCount; root++) {
    if (index[root] !== -1) continue;
    const work = [[root, 0]];
    index[root] = low[root] = counter++; stack.push(root); onStack[root] = 1;
    while (work.length) {
      const frame = work[work.length - 1], v = frame[0];
      if (frame[1] < out[v].length) {
        const w = edges[out[v][frame[1]++]].to;
        if (index[w] === -1) { index[w] = low[w] = counter++; stack.push(w); onStack[w] = 1; work.push([w, 0]); }
        else if (onStack[w]) low[v] = Math.min(low[v], index[w]);
      } else {
        if (low[v] === index[v]) {
          let w; do { w = stack.pop(); onStack[w] = 0; comp[w] = comps; } while (w !== v);
          comps++;
        }
        work.pop();
        if (work.length) { const p = work[work.length - 1][0]; low[p] = Math.min(low[p], low[v]); }
      }
    }
  }
  const sizes = new Array(comps).fill(0);
  for (let i = 0; i < nodeCount; i++) sizes[comp[i]]++;
  let best = 0;
  for (let c = 1; c < comps; c++) if (sizes[c] > sizes[best]) best = c;
  return { comp, components: comps, largest: best, sizes };
}

// Grafın güçlü bağlılık özeti (kapatma/ekleme senaryolarının ağı bölüp bölmediğini görmek için).
export function connectivity(graph) {
  const scc = stronglyConnected(graph.nodes.length, graph.out, graph.edges);
  return { components: scc.components, largestNodes: scc.sizes[scc.largest] ?? 0, connected: scc.components <= 1 };
}

export function subgraph(graph, keepNode) {
  const map = new Int32Array(graph.nodes.length).fill(-1), nodes = [];
  graph.nodes.forEach((n, i) => { if (keepNode(i)) { map[i] = nodes.length; nodes.push({ ...n, index: nodes.length }); } });
  const edges = [];
  for (const e of graph.edges) if (map[e.from] >= 0 && map[e.to] >= 0) edges.push({ ...e, from: map[e.from], to: map[e.to] });
  return finalizeGraph(graph.origin, nodes, edges, graph.report);
}

export function buildRoadGraph(features, options = {}) {
  const { speedMode = 'fixed', fixedSpeed = DEFAULT_SPEED_MS, snapToleranceM = 1.5, pruneToMainComponent = true } = options;
  const roads = [];
  for (const f of features || []) {
    const props = f.properties || {};
    if (props.type && props.type !== 'road') continue;
    if (!f.geometry || !isDrivable(props)) continue;
    const parts = f.geometry.type === 'LineString' ? [f.geometry.coordinates]
      : f.geometry.type === 'MultiLineString' ? f.geometry.coordinates : [];
    parts.forEach((coords, part) => {
      if (coords.length < 2) return;
      const ids = f.geometry.type === 'LineString' && Array.isArray(props.osm_nodes) && props.osm_nodes.length === coords.length ? props.osm_nodes : null;
      roads.push({ id: parts.length > 1 ? `${f.id}#${part}` : String(f.id ?? props.osm_id ?? roads.length), props, coords, ids });
    });
  }
  const empty = { nodes: 0, edges: 0, components: 0, largestNodes: 0, largestNodeShare: 0, largestLengthShare: 0, droppedEdges: 0 };
  if (!roads.length) return finalizeGraph({ lon0: 0, lat0: 0 }, [], [], empty);

  let lonMin = Infinity, lonMax = -Infinity, latMin = Infinity, latMax = -Infinity;
  for (const r of roads) for (const [lon, lat] of r.coords) {
    lonMin = Math.min(lonMin, lon); lonMax = Math.max(lonMax, lon); latMin = Math.min(latMin, lat); latMax = Math.max(latMax, lat);
  }
  const origin = options.origin ?? { lon0: (lonMin + lonMax) / 2, lat0: (latMin + latMax) / 2 };
  const proj = makeProjection(origin.lon0, origin.lat0);

  const uf = new UnionFind();
  const ways = roads.map((r) => ({
    ...r, hasIds: !!r.ids,
    pts: r.coords.map((c) => proj.toXY(c[0], c[1])),
    keys: r.ids ? r.ids.map((n) => `n${n}`) : r.coords.map(coordKey),
  }));
  ways.forEach((w) => w.keys.forEach((k) => uf.find(k)));
  // Aynı koordinatı paylaşan kimliksiz köşeler zaten aynı anahtarı taşır; kesişimleri geometrik bul.
  if (ways.some((w) => !w.hasIds)) connectGeometrically(ways, uf, snapToleranceM);

  const occurrences = new Map();
  for (const w of ways) for (const k of w.keys) { const r = uf.find(k); occurrences.set(r, (occurrences.get(r) || 0) + 1); }

  const nodeIndex = new Map(), nodes = [];
  const nodeOf = (root, pt) => {
    if (!nodeIndex.has(root)) {
      const [lon, lat] = proj.toLonLat(pt[0], pt[1]);
      nodeIndex.set(root, nodes.length);
      nodes.push({ index: nodes.length, id: root, x: pt[0], y: pt[1], lon, lat });
    }
    return nodeIndex.get(root);
  };

  const edges = [];
  for (const w of ways) {
    const dir = onewayDirection(w.props);
    const hw = w.props.highway ?? w.props.class;
    const limit = parseMaxspeed(w.props.maxspeed);
    const speed = speedMode === 'osm' ? (limit ?? fixedSpeed) : fixedSpeed;
    const breaks = [];
    w.keys.forEach((k, i) => { if (i === 0 || i === w.keys.length - 1 || occurrences.get(uf.find(k)) >= 2) breaks.push(i); });
    for (let b = 0; b + 1 < breaks.length; b++) {
      const seg = w.pts.slice(breaks[b], breaks[b + 1] + 1);
      const from = nodeOf(uf.find(w.keys[breaks[b]]), seg[0]);
      const to = nodeOf(uf.find(w.keys[breaks[b + 1]]), seg[seg.length - 1]);
      const length = polylineLength(seg);
      if (from === to || length <= 0) continue;
      const make = (pts, f, t, tag) => ({
        id: `${w.id}:${b}${tag}`, wayId: w.id, osmId: w.props.osm_id ?? null, name: w.props.name ?? null,
        highway: hw, rank: CLASS_RANK[hw] ?? 1, lanes: lanesOf(w.props), from: f, to: t,
        length, speed, freeFlowTime: length / speed, pts, cum: cumulativeLengths(pts),
      });
      if (dir >= 0) edges.push(make(seg, from, to, '+'));
      if (dir <= 0) edges.push(make([...seg].reverse(), to, from, '-'));
    }
  }

  let graph = finalizeGraph(origin, nodes, edges);
  const scc = stronglyConnected(nodes.length, graph.out, edges);
  const inMain = (i) => scc.comp[i] === scc.largest;
  const totalLength = edges.reduce((s, e) => s + e.length, 0);
  const mainLength = edges.reduce((s, e) => s + (inMain(e.from) && inMain(e.to) ? e.length : 0), 0);
  const report = {
    nodes: nodes.length, edges: edges.length, components: scc.components,
    largestNodes: scc.sizes[scc.largest] ?? 0,
    largestNodeShare: nodes.length ? scc.sizes[scc.largest] / nodes.length : 0,
    largestLengthShare: totalLength ? mainLength / totalLength : 0,
    droppedEdges: 0,
  };
  graph.report = report;
  if (pruneToMainComponent && scc.components > 1) {
    graph = subgraph(graph, inMain);
    report.droppedEdges = edges.length - graph.edges.length;
    graph.report = report;
  }
  return graph;
}

// ── Senaryo işlemleri (graf değiştirilmez, yeni graf döner) ──────────────────────────────

// S1: yol kapatma. match: yol (wayId) kümesi veya kenar -> bool fonksiyonu.
export function closeEdges(graph, match) {
  const test = typeof match === 'function' ? match : (e) => (match instanceof Set ? match : new Set(match)).has(e.wayId);
  const edges = graph.edges.filter((e) => !test(e)).map((e) => ({ ...e }));
  const g = finalizeGraph(graph.origin, graph.nodes.map((n) => ({ ...n })), edges, graph.report);
  g.report = { ...graph.report, closedEdges: graph.edges.length - edges.length };
  return g;
}

function nearestNode(graph, lon, lat, maxSnapM) {
  const proj = makeProjection(graph.origin.lon0, graph.origin.lat0);
  const [x, y] = proj.toXY(lon, lat);
  let best = -1, bestD = Infinity;
  graph.nodes.forEach((n, i) => { const d = Math.hypot(n.x - x, n.y - y); if (d < bestD) { bestD = d; best = i; } });
  if (best < 0 || bestD > maxSnapM) throw new Error(`Bağlantı noktası graftaki en yakın düğüme ${bestD.toFixed(0)} m uzakta (sınır ${maxSnapM} m)`);
  return best;
}

// S2: yeni bağlantı yolu. İki uç en yakın mevcut düğümlere bağlanır; düz çizgi kenar eklenir.
export function addConnector(graph, fromLonLat, toLonLat, options = {}) {
  const { twoWay = true, speed = DEFAULT_SPEED_MS, maxSnapM = 60, id = 'connector', highway = 'residential' } = options;
  const a = nearestNode(graph, fromLonLat[0], fromLonLat[1], maxSnapM);
  const b = nearestNode(graph, toLonLat[0], toLonLat[1], maxSnapM);
  if (a === b) throw new Error('Bağlantının iki ucu aynı düğüme düşüyor');
  const na = graph.nodes[a], nb = graph.nodes[b];
  const length = Math.hypot(na.x - nb.x, na.y - nb.y);
  const make = (f, t, p, q, tag) => {
    const pts = [[p.x, p.y], [q.x, q.y]];
    return { id: `${id}:${tag}`, wayId: id, osmId: null, name: null, highway, rank: CLASS_RANK[highway] ?? 1, lanes: 1,
      from: f, to: t, length, speed, freeFlowTime: length / speed, pts, cum: [0, length] };
  };
  const edges = graph.edges.map((e) => ({ ...e }));
  edges.push(make(a, b, na, nb, '+'));
  if (twoWay) edges.push(make(b, a, nb, na, '-'));
  const g = finalizeGraph(graph.origin, graph.nodes.map((n) => ({ ...n })), edges, graph.report);
  g.report = { ...graph.report, addedEdges: twoWay ? 2 : 1, connectorLengthM: length };
  return g;
}
