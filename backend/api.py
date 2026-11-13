"""
api.py -- GeoSim Dijital İkiz Hibrit Mimari API (Mapbox + Roboflow AI + OSM Overpass)
=====================================================================================
Aktif Uç Noktalar:
  POST /analyze        -> Bina ve kentsel nesne segmentasyonu (Roboflow AI + OSM Vektör)
  GET  /api/roads      -> Bounding Box içindeki hazır yol ağları (Overpass API / OSM LineString)
  POST /api/roads      -> Bounding Box içindeki hazır yol ağları (Overpass API / OSM LineString)
  GET  /roboflow-status -> Model durumu
  GET  /model-status    -> Sistem ve harita yapılandırma durumu
"""

import asyncio
import copy
import io
import os
import time
import httpx
from PIL import Image
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Literal
from building_models import scan_buildings, model_info
from prediction_pipeline import PIPELINE_VERSION
from osm_source import fetch_osm_source, supplement_buildings
from regions import router as regions_router
from analysis_archive import router as archive_router, archive_analysis
from references import router as references_router
from vectorize import (
    scan_satellite_image_roboflow,
    _split_and_recover_satellite_roofs,
    roboflow_predictions_to_geojson,
    fetch_mapbox_osm_vectors,
    fetch_osm_roads_overpass,
    merge_hybrid_buildings,
    estimate_osm_offset,
    get_mapbox_bounding_box,
    MAIN_IMAGE_W,
    MAIN_IMAGE_H,
    ROBOFLOW_API_URL,
    ROBOFLOW_API_KEY,
    ROBOFLOW_MODEL_ID,
)

# ── Konfigürasyon (.env dosyasından okunur) ──────────────────────────────────
MAPBOX_ACCESS_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN", "")
MAPBOX_IMAGE_W      = MAIN_IMAGE_W   # 1280 px
MAPBOX_IMAGE_H      = MAIN_IMAGE_H   # 1280 px
MAPBOX_STYLE        = "satellite-v9"

# Model binaları ~0.22 m/piksel çözünürlükte (efektif zoom ~18) iyi buluyor; zoom 17 @1x
# görüntüde (~0.45 m/px) özellikle bitişik nizam binaları kaçırıyordu. Düşük zoom'da
# aynı alanı @2x (2560 px) çekip örtüşmeli 640 px karolarla tarıyoruz.
AI_TARGET_ZOOM  = 18.0
AI_TILE_SIZE    = 640
AI_TILE_OVERLAP = 128

# Kalıcı HTTP istemcisi ve analiz önbelleği
_ASYNC_HTTP_CLIENT: httpx.AsyncClient | None = None
_ANALYZE_CACHE: dict[tuple, tuple[float, dict]] = {}
_SAT_IMAGE_CACHE: dict[tuple, tuple[float, Image.Image]] = {}
_ANALYZE_CACHE_TTL_S = 300.0


def _get_async_client() -> httpx.AsyncClient:
    global _ASYNC_HTTP_CLIENT
    if _ASYNC_HTTP_CLIENT is None or _ASYNC_HTTP_CLIENT.is_closed:
        _ASYNC_HTTP_CLIENT = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=2.5, read=6.0, write=5.0, pool=5.0),
            limits=httpx.Limits(max_keepalive_connections=16, max_connections=32),
        )
    return _ASYNC_HTTP_CLIENT


async def _warmup_default_regions():
    """Açılışta varsayılan harita merkezini ve hazır test bölgelerini arka planda önbelleğe alır."""
    if not MAPBOX_ACCESS_TOKEN or MAPBOX_ACCESS_TOKEN.startswith("TODO"):
        return
    preset_coords = [
        (41.0082, 28.9784, 16.5),
        (41.0295, 28.9485, 17.0),
        (40.9855, 29.0275, 17.0),
        (41.0055, 28.9765, 17.0),
    ]
    for lat, lon, zm in preset_coords:
        try:
            b = get_mapbox_bounding_box(lat, lon, zm, MAPBOX_IMAGE_W, MAPBOX_IMAGE_H)
            await asyncio.to_thread(fetch_mapbox_osm_vectors, b, MAPBOX_ACCESS_TOKEN)
        except Exception:
            pass


# ── Uygulama Lifespan ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("GeoSim API baslatiliyor...")
    print(f"[HAZIR] Aktif Hibrit Bina & Yol Motoru (OSM Vektor + Roboflow AI: {ROBOFLOW_MODEL_ID})")
    warmup_task = asyncio.create_task(_warmup_default_regions())
    yield
    warmup_task.cancel()
    global _ASYNC_HTTP_CLIENT
    if _ASYNC_HTTP_CLIENT is not None and not _ASYNC_HTTP_CLIENT.is_closed:
        await _ASYNC_HTTP_CLIENT.aclose()


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="GeoSim API", version="3.0.0", lifespan=lifespan)
app.include_router(regions_router)
app.include_router(archive_router)
app.include_router(references_router)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request Modelleri ─────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    latitude:   float = Field(ge=-85, le=85)
    longitude:  float = Field(ge=-180, le=180)
    zoom:       float = Field(ge=14, le=20)
    confidence: float = Field(default=0.25, ge=0, le=1)
    model_provider: Literal["roboflow", "geoai", "yolo"] = "roboflow"
    processing_profile: Literal["conservative", "raw", "legacy"] = "conservative"
    include_ai: bool  = True   # False ise yalnızca hızlı OSM/MVT vektör katmanını döndürür


class RoadsRequest(BaseModel):
    south:     float | None = None
    west:      float | None = None
    north:     float | None = None
    east:      float | None = None
    latitude:  float | None = None
    longitude: float | None = None
    zoom:      float = 16.5


# ── /analyze ─────────────────────────────────────────────────────────────────
@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    """
    Hibrit Analiz Endpoint'i (OSM/Mapbox Gerçek Mimari Vektörler + Roboflow AI):
    1. Mapbox Static Images API için Bounding Box (north, south, east, west) sınırlarını hesaplar.
    2. Paralel olarak:
       a) Mapbox/OSM vektör katmanından bölgedeki binaların mimari poligonlarını ve yol ağını çeker.
       b) include_ai=True ise uydu görüntüsünü (düşük zoom'da @2x) örtüşmeli 640x640 karolara
          bölerek Roboflow AI Instance Segmentation modelini çalıştırır ve GeoJSON poligonlarına dönüştürür.
    3. OSM ayak izlerinin zemin konumunu korur, ardından Hibrit Birleştirici ile gerçek
       mimari poligonları ve AI güven skorlarını eşleştirerek deck.gl GeoJsonLayer uyumlu
       standart GeoJSON döndürür.
    """
    if not MAPBOX_ACCESS_TOKEN or MAPBOX_ACCESS_TOKEN.startswith("TODO"):
        raise HTTPException(
            status_code=503,
            detail="Mapbox API token ayarlanmamış. backend/.env içindeki MAPBOX_ACCESS_TOKEN değişkenini kontrol edin."
        )

    t_start = time.perf_counter()
    timings: dict[str, float | list[float] | None] = {}

    scan_zoom = max(16.5, req.zoom) if req.zoom < 15.0 else req.zoom
    scan_zoom = round(scan_zoom, 1)

    selected_model = model_info(req.model_provider)
    cache_key = (req.latitude, req.longitude, scan_zoom, req.confidence, bool(req.include_ai),
                 req.model_provider, req.processing_profile, selected_model.get("weights_sha256"), PIPELINE_VERSION)
    now_mono = time.monotonic()
    cached_resp = _ANALYZE_CACHE.get(cache_key)
    if cached_resp is not None and (now_mono - cached_resp[0]) < _ANALYZE_CACHE_TTL_S:
        return JSONResponse(content=cached_resp[1])

    def _timed(fn):
        """Senkron bir fonksiyonu çalıştığı thread içinde ölçer."""
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                timings[fn.__name__ + "_s"] = round(time.perf_counter() - t0, 3)
        return wrapper

    # 1. 1280x1280 Bounding Box Sınır Koordinatlarını Hesapla
    bounds = get_mapbox_bounding_box(
        center_lat=req.latitude,
        center_lon=req.longitude,
        zoom=scan_zoom,
        img_w=MAPBOX_IMAGE_W,
        img_h=MAPBOX_IMAGE_H,
    )

    # 2. OSM/Mapbox Gerçek Mimari Vektörleri GERÇEK PARALEL Task Olarak Başlat
    real_mapbox = bool(MAPBOX_ACCESS_TOKEN and len(MAPBOX_ACCESS_TOKEN) > 20 and not MAPBOX_ACCESS_TOKEN.startswith("TODO"))
    osm_task = asyncio.create_task(
        asyncio.to_thread(_timed(fetch_mapbox_osm_vectors), bounds, MAPBOX_ACCESS_TOKEN, include_status=True)
    )
    source_task = (
        asyncio.create_task(asyncio.to_thread(fetch_osm_source, bounds, network_timeout=4.0))
        if (req.include_ai or real_mapbox)
        else None
    )

    roboflow_features = []
    satellite_img = None
    roboflow_error = None
    roboflow_res = {}

    # 3. Mapbox Uydu Görüntüsü + Roboflow / Yerel Uydu Çatı Taraması
    scale = 2 if scan_zoom < AI_TARGET_ZOOM - 0.5 else 1
    scan_w, scan_h = MAPBOX_IMAGE_W * scale, MAPBOX_IMAGE_H * scale
    img_w, img_h = scan_w, scan_h
    img_cache_key = (req.latitude, req.longitude, scan_zoom, scale)

    if req.include_ai and selected_model["ready"]:
        cached_img = _SAT_IMAGE_CACHE.get(img_cache_key)
        if cached_img is not None and (time.monotonic() - cached_img[0]) < _ANALYZE_CACHE_TTL_S:
            satellite_img = cached_img[1].copy()
            timings["mapbox_image_s"] = 0.0
        else:
            mapbox_url = (
                f"https://api.mapbox.com/styles/v1/mapbox/{MAPBOX_STYLE}/static/"
                f"{req.longitude},{req.latitude},{scan_zoom:.1f},0/"
                f"{MAPBOX_IMAGE_W}x{MAPBOX_IMAGE_H}{'@2x' if scale == 2 else ''}"
                f"?access_token={MAPBOX_ACCESS_TOKEN}"
            )
            t0 = time.perf_counter()
            try:
                client = _get_async_client()
                resp = await client.get(mapbox_url)
                if resp.status_code == 200:
                    satellite_img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                    if satellite_img.size != (scan_w, scan_h):
                        satellite_img = satellite_img.resize((scan_w, scan_h), Image.Resampling.LANCZOS)
                    _SAT_IMAGE_CACHE[img_cache_key] = (time.monotonic(), satellite_img)
                elif req.include_ai:
                    roboflow_error = f"Uydu görüntüsü alınamadı (HTTP {resp.status_code})."
                timings["mapbox_image_s"] = round(time.perf_counter() - t0, 3)
            except Exception as img_err:
                if req.include_ai:
                    roboflow_error = f"Uydu görüntüsü alınamadı ({type(img_err).__name__})."
                timings["mapbox_image_s"] = round(time.perf_counter() - t0, 3)
                print(f"[UYARI] Mapbox static goruntu hatasi: {img_err}")

        if satellite_img is not None:
            if satellite_img.size != (scan_w, scan_h):
                satellite_img = satellite_img.resize((scan_w, scan_h), Image.Resampling.LANCZOS)
            img_w, img_h = satellite_img.size
            if req.include_ai and selected_model["ready"]:
                try:
                    t0 = time.perf_counter()
                    roboflow_res = await asyncio.to_thread(
                        _timed(scan_buildings),
                        satellite_img,
                        provider=req.model_provider,
                        processing_profile=req.processing_profile,
                        confidence_threshold=req.confidence,
                        tile_size=AI_TILE_SIZE,
                        overlap=AI_TILE_OVERLAP,
                        target_size=(scan_w, scan_h),
                    )
                    timings["roboflow_total_s"] = round(time.perf_counter() - t0, 3)
                    timings["roboflow_tiles_s"] = roboflow_res.get("tile_timings_s")
                except Exception as rf_e:
                    roboflow_error = str(rf_e)
                    print(f"[UYARI] Roboflow cikarim hatasi: {rf_e}")
    if req.include_ai and not selected_model["ready"]:
        roboflow_error = selected_model.get("reason") or "Model is not configured."

    # 4. Özgün OSM geometrisi, Mapbox karolarındaki binalara göre önceliklidir.
    osm_buildings, osm_roads = [], []
    mvt_status = {"complete": False, "tiles": 0, "failed_tiles": 0}
    t0 = time.perf_counter()
    try:
        osm_buildings, osm_roads, mvt_status = await osm_task
    except Exception as osm_err:
        print(f"[UYARI] Vektor bina katmani hatasi: {osm_err}")
    timings["osm_wait_s"] = round(time.perf_counter() - t0, 3)
    source_result = await source_task if source_task else None
    if source_result:
        osm_buildings = supplement_buildings(source_result["buildings"], osm_buildings)
        if source_result["roads"]:
            osm_roads = source_result["roads"]

    if roboflow_res:
        try:
            t0 = time.perf_counter()
            rf_geojson = roboflow_predictions_to_geojson(
                predictions_or_result=roboflow_res,
                img_w=img_w,
                img_h=img_h,
                center_lat=req.latitude,
                center_lon=req.longitude,
                zoom=scan_zoom,
                bounds=bounds,
                default_height=15.0,
                simplify_epsilon=1.5 * scale,
                osm_roads=osm_roads,
            )
            roboflow_features = rf_geojson.get("features", [])
            timings["roboflow_to_geojson_s"] = round(time.perf_counter() - t0, 3)
        except Exception as rf_e:
            roboflow_error = str(rf_e)
            print(f"[UYARI] Roboflow GeoJSON donusum hatasi: {rf_e}")

    t0 = time.perf_counter()
    osm_offset = None
    # Yalnızca OSM sonucu, hibrit birleştirmeden etkilenmesin diye kopyalanır.
    osm_baseline = copy.deepcopy(osm_buildings + osm_roads)
    if roboflow_features:
        # Çatı kayması ölçülür; simülasyonda kullanılan zemin ayak izleri kaydırılmaz.
        osm_offset = estimate_osm_offset(osm_buildings, roboflow_features, bounds)
        hybrid_buildings = merge_hybrid_buildings(
            osm_buildings,
            copy.deepcopy(roboflow_features),
            min_gapfill_conf=req.confidence,
            osm_roads=osm_roads,
            bounds=bounds,
        )
    else:
        hybrid_buildings = list(osm_buildings)
    timings["merge_s"] = round(time.perf_counter() - t0, 3)
    building_count = len(hybrid_buildings)
    final_features = list(hybrid_buildings)

    if osm_roads:
        final_features.extend(osm_roads)

    timings["total_s"] = round(time.perf_counter() - t_start, 3)

    payload = {
        "type": "FeatureCollection",
        "features": final_features,
        "meta": {
            "source": "hybrid_osm_" + req.model_provider if (osm_buildings and roboflow_features) else ("osm_mvt" if osm_buildings else (req.model_provider if building_count > 0 else "none")),
            "model_id": selected_model["model_id"],
            "model_provider": req.model_provider,
            "weights_sha256": selected_model.get("weights_sha256"),
            "processing_profile": req.processing_profile,
            "pipeline_version": PIPELINE_VERSION,
            "trained_images": None,
            "detected_buildings": building_count,
            "ai_raw_detections": len(roboflow_res.get("raw_predictions", [])),
            "ai_processed_detections": len(roboflow_features),
            "osm_vector_buildings": len(osm_buildings),
            "osm_original_buildings": len(source_result["buildings"]) if source_result else 0,
            "osm_source_error": source_result["error"] if source_result else None,
            "osm_timestamp": source_result["timestamp"] if source_result else None,
            "osm_from_snapshot": source_result.get("from_snapshot", False) if source_result else False,
            "osm_snapshot_sha256": source_result.get("snapshot_sha256") if source_result else None,
            "osm_mvt_status": mvt_status,
            "osm_input_complete": bool((source_result and not source_result["error"]) or mvt_status["complete"]),
            "osm_layer_source": "original_osm" if source_result and not source_result["error"] else "mapbox_mvt",
            "coverage_verified": False,
            "review_status": "unreviewed",
            "osm_offset_m": {"east": osm_offset[2], "north": osm_offset[3]} if osm_offset else None,
            "osm_offset_applied": False,
            "ai_scan_scale": scale if req.include_ai else None,
            "total_features": len(final_features),
            "roboflow_error": roboflow_error,
            "center": {"lat": req.latitude, "lon": req.longitude},
            "zoom": req.zoom,
            "scan_zoom": scan_zoom,
            "bounds": bounds,
            "tiles": roboflow_res.get("tiles", 0),
            "tile_size": AI_TILE_SIZE,
            "tile_overlap": AI_TILE_OVERLAP,
            "failed_tiles": roboflow_res.get("failed_tiles", 0),
            "timings": timings,
        }
    }

    if req.include_ai:
        try:
            manifest = await asyncio.to_thread(archive_analysis, request=req.model_dump(), payload=payload,
                image=satellite_img, osm_features=osm_baseline, ai_features=roboflow_features,
                model_output=roboflow_res)
            payload["meta"]["analysis_run_id"] = manifest["id"]
            payload["meta"]["archive_status"] = manifest["status"]
        except (OSError, ValueError) as archive_error:
            payload["meta"]["archive_error"] = type(archive_error).__name__
        timings["total_s"] = round(time.perf_counter() - t_start, 3)

    # Eksik (karo düşmüş / Roboflow hatalı) sonucu cache'leme; bir sonraki istek yeniden denesin
    if (not roboflow_error and not roboflow_res.get("failed_tiles") and payload["meta"]["osm_input_complete"]
            and not payload["meta"].get("archive_error")):
        if len(_ANALYZE_CACHE) > 64:
            _ANALYZE_CACHE.clear()
        _ANALYZE_CACHE[cache_key] = (now_mono, payload)

    return JSONResponse(content=payload)


# ── /api/roads (Overpass API / MVT Yol Ağları Endpoint'i) ────────────────────
def _resolve_road_bounds(
    south: float | None = None,
    west: float | None = None,
    north: float | None = None,
    east: float | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    zoom: float = 16.5,
) -> dict:
    """
    Doğrudan verilen Bounding Box (south, west, north, east) koordinatlarını
    veya merkez (latitude, longitude, zoom) bilgisinden hesaplanan Bounding Box'ı döndürür.
    """
    if south is not None and west is not None and north is not None and east is not None:
        return {
            "south": float(south),
            "west":  float(west),
            "north": float(north),
            "east":  float(east),
        }
    if latitude is not None and longitude is not None:
        scan_zoom = max(16.0, float(zoom)) if float(zoom) < 15.0 else float(zoom)
        return get_mapbox_bounding_box(
            center_lat=float(latitude),
            center_lon=float(longitude),
            zoom=round(scan_zoom, 1),
            img_w=MAPBOX_IMAGE_W,
            img_h=MAPBOX_IMAGE_H,
        )
    raise HTTPException(
        status_code=400,
        detail="Bounding Box (south, west, north, east) veya merkez koordinatları (latitude, longitude) gereklidir."
    )


@app.get("/api/roads")
async def get_roads(
    south: float | None = None,
    west: float | None = None,
    north: float | None = None,
    east: float | None = None,
    lat: float | None = None,
    lon: float | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    zoom: float = 16.5,
):
    """
    Ekranda görüntülenen Bounding Box koordinatlarını alarak o bölgedeki yol ağlarını
    (highways, streets) çeker ve standart GeoJSON (LineString) FeatureCollection formatında döndürür.
    """
    c_lat = latitude if latitude is not None else lat
    c_lon = longitude if longitude is not None else lon
    bounds = _resolve_road_bounds(south, west, north, east, c_lat, c_lon, zoom)
    road_geojson = await asyncio.to_thread(fetch_osm_roads_overpass, bounds, MAPBOX_ACCESS_TOKEN)
    return JSONResponse(content=road_geojson)


@app.post("/api/roads")
async def post_roads(req: RoadsRequest):
    """
    POST ile gönderilen Bounding Box veya merkez koordinatları için
    yol ağlarını (LineString GeoJSON) döndürür.
    """
    bounds = _resolve_road_bounds(
        req.south, req.west, req.north, req.east,
        req.latitude, req.longitude, req.zoom
    )
    road_geojson = await asyncio.to_thread(fetch_osm_roads_overpass, bounds, MAPBOX_ACCESS_TOKEN)
    return JSONResponse(content=road_geojson)


# ── /api/search (Konum / Bölge Arama Endpoint'i) ─────────────────────────────
_SEARCH_CACHE: dict[str, tuple[float, list[dict]]] = {}


@app.get("/api/search")
async def search_location(q: str = "", limit: int = 6):
    """
    Aranan yer, ilçe, mahalle, cadde veya koordinat metnini Mapbox Geocoding API
    (ve yedek olarak OpenStreetMap Nominatim) üzerinden sorgular ve eşleşen
    konumların koordinat listesini döndürür.
    """
    import urllib.parse

    query = (q or "").strip()
    if len(query) < 2:
        return {"results": []}

    cache_key = f"{query.lower()}:{limit}"
    now_mono = time.monotonic()
    cached = _SEARCH_CACHE.get(cache_key)
    if cached is not None and (now_mono - cached[0]) < 600.0:
        return {"results": cached[1]}

    results: list[dict] = []

    # 1. Doğrudan koordinat girildiyse (örn: "41.0082, 28.9784")
    parts = [p.strip() for p in query.replace(";", ",").split(",") if p.strip()]
    if len(parts) == 2:
        try:
            c_lat = float(parts[0])
            c_lon = float(parts[1])
            if -90.0 <= c_lat <= 90.0 and -180.0 <= c_lon <= 180.0:
                results.append({
                    "id": f"coord_{c_lat}_{c_lon}",
                    "name": f"{c_lat:.5f}, {c_lon:.5f}",
                    "full_name": f"Koordinat: {c_lat:.6f}, {c_lon:.6f}",
                    "lat": c_lat,
                    "lon": c_lon,
                    "zoom": 17.0,
                })
        except ValueError:
            pass

    client = _get_async_client()

    # 2. Mapbox Geocoding API
    if MAPBOX_ACCESS_TOKEN and not MAPBOX_ACCESS_TOKEN.startswith("TODO"):
        try:
            encoded_q = urllib.parse.quote(query)
            mb_url = (
                f"https://api.mapbox.com/geocoding/v5/mapbox.places/{encoded_q}.json"
                f"?access_token={MAPBOX_ACCESS_TOKEN}&language=tr&limit={limit}"
                f"&types=poi,address,neighborhood,locality,place,district,region"
            )
            resp = await client.get(mb_url)
            if resp.status_code == 200:
                data = resp.json()
                for idx, feat in enumerate(data.get("features", [])):
                    center = feat.get("center")
                    if not center or len(center) < 2:
                        continue
                    lon, lat = float(center[0]), float(center[1])
                    results.append({
                        "id": feat.get("id") or f"mb_{idx}",
                        "name": feat.get("text_tr") or feat.get("text") or query,
                        "full_name": feat.get("place_name_tr") or feat.get("place_name") or query,
                        "lat": round(lat, 6),
                        "lon": round(lon, 6),
                        "zoom": 17.0,
                    })
        except Exception as mb_err:
            print(f"[UYARI] Mapbox arama hatasi: {mb_err}")

    # 3. Yedek: OpenStreetMap Nominatim API
    if not results:
        try:
            nom_url = "https://nominatim.openstreetmap.org/search"
            resp = await client.get(
                nom_url,
                params={"q": query, "format": "json", "limit": limit, "accept-language": "tr"},
                headers={"User-Agent": "GeoSim-DigitalTwin/3.0"},
            )
            if resp.status_code == 200:
                for idx, item in enumerate(resp.json()):
                    lat = float(item["lat"])
                    lon = float(item["lon"])
                    disp = item.get("display_name", query)
                    short_name = disp.split(",")[0].strip()
                    results.append({
                        "id": f"nom_{item.get('place_id', idx)}",
                        "name": short_name,
                        "full_name": disp,
                        "lat": round(lat, 6),
                        "lon": round(lon, 6),
                        "zoom": 17.0,
                    })
        except Exception as nom_err:
            print(f"[UYARI] Nominatim arama hatasi: {nom_err}")

    if len(_SEARCH_CACHE) > 128:
        _SEARCH_CACHE.clear()
    _SEARCH_CACHE[cache_key] = (now_mono, results)
    return {"results": results}


# ── /roboflow-status & /model-status ──────────────────────────────────────────
@app.get("/roboflow-status")
async def roboflow_status():
    """Roboflow Instance Segmentation model durumunu döndürür."""
    return {
        "status": "configured" if ROBOFLOW_API_KEY else "unconfigured",
        "model_id": ROBOFLOW_MODEL_ID,
        "api_url": ROBOFLOW_API_URL,
        "api_key_configured": bool(ROBOFLOW_API_KEY),
        "trained_images": None,
        "task": "instance_segmentation",
        "class": "building",
        "default_height": 15.0,
    }


@app.get("/model-status")
async def model_status():
    """Sistem ve model hazırlık durumunu döndürür."""
    return {
        "providers": [model_info(name) for name in ("roboflow", "geoai", "yolo")],
        "default_processing_profile": "conservative",
        "pipeline_version": PIPELINE_VERSION,
        "mapbox_configured": bool(MAPBOX_ACCESS_TOKEN) and not MAPBOX_ACCESS_TOKEN.startswith("TODO"),
        "roboflow": {
            "ready": bool(ROBOFLOW_API_KEY),
            "model_id": ROBOFLOW_MODEL_ID,
            "api_url": ROBOFLOW_API_URL,
            "trained_images": None,
            "task": "instance_segmentation",
        },
    }
