// Bina alanı × kat sayısına dayalı düğüm talep ağırlıkları:  w_i = w_0 + Σ Alan(b) × Kat(b)
// toplam, düğüme `radiusM` (75 m) içinde kalan binalar üzerinden alınır.
//
// Kat sayısı yalnız OSM'de açıkça varsa kullanılır; yoksa (yapay zekâ binaları dahil) mahalle medyanı
// uygulanır. `floorsMode: 'one'` tüm binalarda kat = 1 alan duyarlılık analizidir.
import { makeProjection, ringAreaCentroid } from './geo.js';

function outerRings(geometry) {
  if (!geometry) return [];
  if (geometry.type === 'Polygon') return [geometry.coordinates[0]];
  if (geometry.type === 'MultiPolygon') return geometry.coordinates.map((p) => p[0]);
  return [];
}

function median(values) {
  if (!values.length) return null;
  const s = [...values].sort((a, b) => a - b), m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

export function buildingRecords(buildings, graph) {
  const proj = makeProjection(graph.origin.lon0, graph.origin.lat0);
  const records = [];
  for (const f of buildings || []) {
    const props = f.properties || {};
    if (props.type && props.type !== 'building') continue;
    let area = 0, cx = 0, cy = 0;
    for (const ring of outerRings(f.geometry)) {
      const r = ring.map((c) => proj.toXY(c[0], c[1]));
      const a = ringAreaCentroid(r);
      if (a.area > area) { area = a.area; cx = a.cx; cy = a.cy; }
    }
    if (area <= 0) continue;
    const floors = Number(props.floors);
    const osmFloors = (props.source === 'osm' || props.geometry_source === 'overpass') && Number.isFinite(floors) && floors > 0 ? floors : null;
    records.push({ id: f.id, area, x: cx, y: cy, osmFloors });
  }
  return records;
}

export function nodeWeights(graph, buildings, options = {}) {
  const { radiusM = 75, baseWeight = 1, floorsMode = 'median', onlyNodes = null } = options;
  const records = buildingRecords(buildings, graph);
  const medianFloors = floorsMode === 'one' ? 1 : (median(records.filter((r) => r.osmFloors).map((r) => r.osmFloors)) ?? 1);
  const cell = radiusM, grid = new Map();
  graph.nodes.forEach((n, i) => {
    const k = `${Math.floor(n.x / cell)},${Math.floor(n.y / cell)}`;
    if (!grid.has(k)) grid.set(k, []); grid.get(k).push(i);
  });
  const weights = new Float64Array(graph.nodes.length).fill(baseWeight);
  let covered = 0, coveredFloorArea = 0, totalFloorArea = 0;
  for (const b of records) {
    const floors = floorsMode === 'one' ? 1 : (b.osmFloors ?? medianFloors);
    const floorArea = b.area * floors;
    totalFloorArea += floorArea;
    const gx = Math.floor(b.x / cell), gy = Math.floor(b.y / cell);
    let hit = false;
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {
      for (const i of grid.get(`${gx + dx},${gy + dy}`) || []) {
        const n = graph.nodes[i];
        if (Math.hypot(n.x - b.x, n.y - b.y) <= radiusM) { weights[i] += floorArea; hit = true; }
      }
    }
    if (hit) { covered++; coveredFloorArea += floorArea; }
  }
  if (onlyNodes) for (let i = 0; i < weights.length; i++) if (!onlyNodes.has(i)) weights[i] = 0;
  return {
    weights, medianFloors, floorsMode, radiusM, baseWeight,
    buildings: records.length, buildingsWithinRadius: covered,
    buildingCoverage: records.length ? covered / records.length : 0,
    floorAreaCoverage: totalFloorArea ? coveredFloorArea / totalFloorArea : 0,
  };
}

// Ağırlıklarla orantılı düğüm seçimi (kümülatif toplam + ikili arama).
export function makeSampler(weights) {
  const cum = new Float64Array(weights.length);
  let total = 0;
  for (let i = 0; i < weights.length; i++) { total += weights[i]; cum[i] = total; }
  if (!(total > 0)) throw new Error('Talep ağırlıklarının toplamı sıfır');
  return (u) => {
    const target = u * total;
    let lo = 0, hi = cum.length - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (cum[mid] > target) hi = mid; else lo = mid + 1; }
    return lo;
  };
}
