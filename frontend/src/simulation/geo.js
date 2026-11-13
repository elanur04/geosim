// Küçük alanlar için eşdikdörtgen (equirectangular) yerel projeksiyon: [boylam, enlem] -> metre.
const EARTH_RADIUS_M = 6371008.8;
const RAD = Math.PI / 180;

export function makeProjection(lon0, lat0) {
  const kx = EARTH_RADIUS_M * RAD * Math.cos(lat0 * RAD);
  const ky = EARTH_RADIUS_M * RAD;
  return {
    lon0, lat0,
    toXY: (lon, lat) => [(lon - lon0) * kx, (lat - lat0) * ky],
    toLonLat: (x, y) => [lon0 + x / kx, lat0 + y / ky],
  };
}

export function polylineLength(points) {
  let total = 0;
  for (let i = 1; i < points.length; i++) total += Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]);
  return total;
}

export function distancePointToSegment(px, py, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay;
  const len2 = dx * dx + dy * dy;
  const t = len2 === 0 ? 0 : Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len2));
  return { distance: Math.hypot(px - (ax + t * dx), py - (ay + t * dy)), t };
}

// Poligon halkasının alanı ve ağırlık merkezi (metre cinsinden, işaretsiz alan).
export function ringAreaCentroid(ring) {
  let twiceArea = 0, cx = 0, cy = 0;
  for (let i = 0, n = ring.length; i < n; i++) {
    const [x0, y0] = ring[i], [x1, y1] = ring[(i + 1) % n];
    const cross = x0 * y1 - x1 * y0;
    twiceArea += cross; cx += (x0 + x1) * cross; cy += (y0 + y1) * cross;
  }
  if (Math.abs(twiceArea) < 1e-9) return { area: 0, cx: ring[0]?.[0] ?? 0, cy: ring[0]?.[1] ?? 0 };
  return { area: Math.abs(twiceArea) / 2, cx: cx / (3 * twiceArea), cy: cy / (3 * twiceArea) };
}

// Polilinin başından s metre ilerideki nokta ve yön (radyan).
export function pointAlong(points, cumulative, s) {
  const total = cumulative[cumulative.length - 1];
  const d = Math.max(0, Math.min(total, s));
  let i = 1;
  while (i < cumulative.length - 1 && cumulative[i] < d) i++;
  const seg = cumulative[i] - cumulative[i - 1] || 1;
  const t = (d - cumulative[i - 1]) / seg;
  const [ax, ay] = points[i - 1], [bx, by] = points[i];
  return { x: ax + (bx - ax) * t, y: ay + (by - ay) * t, heading: Math.atan2(by - ay, bx - ax) };
}
