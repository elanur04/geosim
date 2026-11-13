"""Overpass üzerinden özgün OSM bina ve yol geometrilerini çeker (Mapbox genelleştirmesinden bağımsız)."""
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import requests
from shapely.geometry import LineString, Polygon, box, mapping, shape
from shapely.ops import polygonize, unary_union
from shapely.strtree import STRtree

# Kullanılan Overpass API sunucuları.
ENDPOINTS = ("https://overpass.private.coffee/api/interpreter", "https://overpass-api.de/api/interpreter")
_CACHE = {}
SOURCE_ROOT = Path(__file__).resolve().parent / "data" / "osm_sources"
SEED_SOURCE = Path(__file__).resolve().parent / "verification" / "bartin-source.geojson"


def _snapshot(bounds):
    """Sabit OSM kayıtlarını yeniden kullanır; kullanıcının düzenlediği çalışma alanlarını kullanmaz."""
    from shapely.geometry import shape
    candidates = ([SEED_SOURCE] if SEED_SOURCE.exists() else [])
    if SOURCE_ROOT.exists():
        candidates += list(SOURCE_ROOT.glob('*.geojson'))
    extent = box(bounds['west'], bounds['south'], bounds['east'], bounds['north'])
    for path in candidates:
        try:
            raw = path.read_bytes()
            data = json.loads(raw)
            meta = data['meta']
            b = meta['bounds']
            if not (b['west'] <= bounds['west'] < bounds['east'] <= b['east'] and
                    b['south'] <= bounds['south'] < bounds['north'] <= b['north']):
                continue
            features = [f for f in data['features'] if shape(f['geometry']).intersects(extent)]
            return {'buildings': [f for f in features if f['properties']['type'] == 'building'],
                    'roads': [f for f in features if f['properties']['type'] == 'road'],
                    'error': None, 'timestamp': meta.get('osm_timestamp'), 'from_snapshot': True,
                    'snapshot_sha256': hashlib.sha256(raw).hexdigest()}
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def _save_source(bounds, result):
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    raw = json.dumps({'type': 'FeatureCollection', 'features': result['buildings'] + result['roads'],
                      'meta': {'bounds': bounds, 'osm_timestamp': result['timestamp']}}, ensure_ascii=False).encode('utf-8')
    path = SOURCE_ROOT / f"{hashlib.sha256(raw).hexdigest()}.geojson"
    # İçerik özetiyle adlandırılan dosyalar değişmez; çevrimdışı tekrar kullanılabilir.
    with path.open('xb') as output:
        output.write(raw)


def _number(value):
    try:
        result = float(str(value).removesuffix(" m").replace(",", "."))
        return result if math.isfinite(result) and result > 0 else None
    except (ValueError, TypeError):
        return None


def parse_osm(data, bounds):
    """Tam halkaları, avluları, OSM kimliklerini ve yol düğüm bağlantılarını korur."""
    extent = box(bounds["west"], bounds["south"], bounds["east"], bounds["north"])
    scale = 111320 ** 2 * math.cos(math.radians((bounds["north"] + bounds["south"]) / 2))
    buildings, roads = [], []
    elements = data.get("elements", [])
    member_ids = set()

    def coords(geometry):
        return [(p["lon"], p["lat"]) for p in geometry or [] if "lon" in p and "lat" in p]

    for el in sorted(elements, key=lambda e: e.get("type") != "relation"):
        tags = el.get("tags") or {}
        points = coords(el.get("geometry"))
        key = f"osm_{el['type']}_{el['id']}"
        if tags.get("highway") and tags.get("area") != "yes" and len(points) >= 2:
            road = LineString(points)
            if road.intersects(extent):
                roads.append({"type": "Feature", "id": key, "geometry": mapping(road), "properties": {
                    "type": "road", "source": "osm", "geometry_source": "overpass", "osm_id": el["id"],
                    "osm_nodes": el.get("nodes", []), "class": tags["highway"], "highway": tags["highway"],
                    "name": tags.get("name"), "oneway": tags.get("oneway", "no"),
                    "bridge": tags.get("bridge", "no"), "tunnel": tags.get("tunnel", "no"),
                    "layer": tags.get("layer", "0"), "access": tags.get("access"),
                    "lanes": tags.get("lanes"), "maxspeed": tags.get("maxspeed"),
                    "junction": tags.get("junction"),  # rotari = tek yön (trafik simülasyonu için)
                    "width": _number(tags.get("width")) or 3, "color": [255, 140, 0],
                }})
        if (not tags.get("building") or tags["building"] == "no"
                or (el["type"] == "way" and el["id"] in member_ids)):
            continue
        geom = None
        if el["type"] == "relation":
            members = el.get("members", [])
            outer, inner = [], []
            for member in members:
                pts = coords(member.get("geometry"))
                if len(pts) >= 2:
                    (inner if member.get("role") == "inner" else outer).append(LineString(pts))
            shells = list(polygonize(unary_union(outer))) if outer else []
            holes = list(polygonize(unary_union(inner))) if inner else []
            if shells:
                geom = unary_union(shells)
                if holes:
                    geom = geom.difference(unary_union(holes))
        elif len(points) >= 4 and points[0] == points[-1]:
            geom = Polygon(points)
        if geom is None or geom.is_empty or not geom.is_valid or not geom.intersects(extent):
            continue
        if el["type"] == "relation":
            member_ids.update(m["ref"] for m in el.get("members", []) if m.get("type") == "way")
        height = _number(tags.get("height"))
        levels = _number(tags.get("building:levels"))
        buildings.append({"type": "Feature", "id": key, "geometry": mapping(geom), "properties": {
            "type": "building", "class": "Building", "source": "osm", "geometry_source": "overpass",
            "osm_id": el["id"], "osm_type": el["type"], "name": tags.get("name") or "Bina",
            "building": tags["building"], "height": height or (levels * 3 if levels else 9),
            "height_source": "osm" if height else ("levels_estimate" if levels else "display_default"),
            "floors": levels, "area_m2": round(geom.area * scale, 1), "elevation": 0,
            "color": [0, 255, 157], "review_status": "unreviewed",
        }})
    return buildings, roads


def fetch_osm_source(bounds, refresh=False, network_timeout=35.0):
    key = tuple(float(bounds[k]) for k in ("south", "west", "north", "east"))
    cached = _CACHE.get(key)
    if not refresh and cached and time.monotonic() - cached[0] < 600:
        return copy.deepcopy(cached[1])
    stored = _snapshot(bounds) if not refresh else None
    if stored is not None:
        _CACHE[key] = (time.monotonic(), copy.deepcopy(stored))
        return stored
    south, west, north, east = key
    bbox = f"({south},{west},{north},{east})"
    query_timeout = max(1, min(25, int(network_timeout)))
    query = f'[out:json][timeout:{query_timeout}];(way["building"]{bbox};relation["building"]["type"="multipolygon"]{bbox};way["highway"]{bbox};);out body geom;'
    errors = []
    for url in ENDPOINTS:
        try:
            response = requests.get(url, params={"data": query}, timeout=(min(2, network_timeout), network_timeout),
                                     headers={"User-Agent": "GeoSim/3.0"})
            response.raise_for_status()
            data = response.json()
            if data.get("remark") or "elements" not in data:
                raise ValueError(data.get("remark") or "Missing elements")
            buildings, roads = parse_osm(data, bounds)
            result = {"buildings": buildings, "roads": roads, "error": None,
                      "timestamp": data.get("osm3s", {}).get("timestamp_osm_base"), "from_snapshot": False}
            try:
                _save_source(bounds, result)
            except OSError:
                pass  # Diske yazılamasa da ağdan alınan veri kullanılmaya devam eder.
            if len(_CACHE) >= 64:
                _CACHE.clear()
            _CACHE[key] = (time.monotonic(), copy.deepcopy(result))
            return result
        except (requests.RequestException, ValueError, KeyError) as exc:
            errors.append(f"{url}: {type(exc).__name__}")
    return {"buildings": [], "roads": [], "error": "; ".join(errors), "timestamp": None}


def supplement_buildings(original, fallback):
    """Özgün OSM geometrisini temel alır; yalnız Mapbox'ta olan ve çakışmayan binaları ekler."""
    result = list(original)
    if not original:
        return list(fallback)
    geoms = [shape(f["geometry"]) for f in original]
    tree = STRtree(geoms)
    for feature in fallback:
        geom = shape(feature["geometry"])
        if not geom.is_valid or geom.is_empty:
            continue
        if any(geom.intersection(geoms[int(i)]).area / max(min(geom.area, geoms[int(i)].area), 1e-15) > 0.3
               for i in tree.query(geom)):
            continue
        result.append(feature)
    return result
