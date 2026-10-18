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
import io
import os
import httpx
from PIL import Image
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from vectorize import (
    scan_satellite_image_roboflow,
    roboflow_predictions_to_geojson,
    fetch_mapbox_osm_vectors,
    fetch_osm_roads_overpass,
    merge_hybrid_buildings,
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


# ── Uygulama Lifespan ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("GeoSim API başlatılıyor...")
    print(f"✅ Aktif Hibrit Bina & Yol Motoru (OSM Vektör + Roboflow AI: {ROBOFLOW_MODEL_ID}) [HAZIR]")
    yield


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="GeoSim API", version="3.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request Modelleri ─────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    latitude:   float          # Harita merkezi (enlem)
    longitude:  float          # Harita merkezi (boylam)
    zoom:       float          # Mapbox zoom seviyesi (genellikle 15-18)
    confidence: float = 0.25   # Minimum model güveni


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
       b) 1280x1280 uydu görüntüsünü 2x2 (4 çeyrek: 640x640) parçaya bölerek Roboflow AI
          Instance Segmentation modelini çalıştırır ve GeoJSON poligonlarına dönüştürür.
    3. Hibrit Birleştirici ile gerçek mimari poligonları ve AI güven skorlarını eşleştirerek
       deck.gl GeoJsonLayer uyumlu standart GeoJSON döndürür.
    """
    if not MAPBOX_ACCESS_TOKEN or MAPBOX_ACCESS_TOKEN.startswith("TODO"):
        raise HTTPException(
            status_code=503,
            detail="Mapbox API token ayarlanmamış. backend/.env içindeki MAPBOX_ACCESS_TOKEN değişkenini kontrol edin."
        )

    scan_zoom = max(16.5, req.zoom) if req.zoom < 15.0 else req.zoom
    scan_zoom = round(scan_zoom, 1)

    # 1. 1280x1280 Bounding Box Sınır Koordinatlarını Hesapla
    bounds = get_mapbox_bounding_box(
        center_lat=req.latitude,
        center_lon=req.longitude,
        zoom=scan_zoom,
        img_w=MAPBOX_IMAGE_W,
        img_h=MAPBOX_IMAGE_H,
    )

    # 2. OSM/Mapbox Gerçek Mimari Vektörleri Paralel Başlat
    osm_task = asyncio.to_thread(fetch_mapbox_osm_vectors, bounds, MAPBOX_ACCESS_TOKEN)

    # 3. Mapbox 1280x1280 Uydu Görüntüsü + Roboflow 2x2 Tiling
    mapbox_url = (
        f"https://api.mapbox.com/styles/v1/mapbox/{MAPBOX_STYLE}/static/"
        f"{req.longitude},{req.latitude},{scan_zoom:.1f},0/"
        f"{MAPBOX_IMAGE_W}x{MAPBOX_IMAGE_H}"
        f"?access_token={MAPBOX_ACCESS_TOKEN}"
    )

    satellite_img = None
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(mapbox_url)
            if resp.status_code == 200:
                satellite_img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        try:
            import requests
            r = requests.get(mapbox_url, timeout=15)
            if r.status_code == 200:
                satellite_img = Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception:
            satellite_img = None

    roboflow_features = []
    roboflow_error = None
    roboflow_res = {}

    if satellite_img is not None:
        if satellite_img.size != (MAPBOX_IMAGE_W, MAPBOX_IMAGE_H):
            satellite_img = satellite_img.resize((MAPBOX_IMAGE_W, MAPBOX_IMAGE_H), Image.Resampling.LANCZOS)
        img_w, img_h = satellite_img.size
        try:
            roboflow_res = await asyncio.to_thread(
                scan_satellite_image_roboflow,
                satellite_img,
                ROBOFLOW_API_URL,
                ROBOFLOW_API_KEY,
                ROBOFLOW_MODEL_ID,
                True,
                req.confidence,
                None,
                0,
                4,
                (MAPBOX_IMAGE_W, MAPBOX_IMAGE_H),
            )
            rf_geojson = roboflow_predictions_to_geojson(
                predictions_or_result=roboflow_res,
                img_w=img_w,
                img_h=img_h,
                center_lat=req.latitude,
                center_lon=req.longitude,
                zoom=scan_zoom,
                bounds=bounds,
                default_height=15.0,
                simplify_epsilon=1.5,
            )
            roboflow_features = rf_geojson.get("features", [])
        except Exception as rf_e:
            roboflow_error = str(rf_e)
            print(f"⚠️ Roboflow çıkarım hatası: {rf_e}")

    # 4. Hibrit Birleştirme (OSM Gerçek Mimari Vektörler + Roboflow AI)
    osm_buildings, osm_roads = [], []
    try:
        osm_buildings, osm_roads = await osm_task
    except Exception as osm_err:
        print(f"⚠️ Vektör bina katmanı hatası: {osm_err}")

    hybrid_buildings = merge_hybrid_buildings(osm_buildings, roboflow_features)
    building_count = len(hybrid_buildings)
    final_features = list(hybrid_buildings)

    if osm_roads:
        final_features.extend(osm_roads)

    return JSONResponse(content={
        "type": "FeatureCollection",
        "features": final_features,
        "meta": {
            "source": "hybrid_osm_roboflow" if osm_buildings else ("roboflow" if building_count > 0 else "none"),
            "model_id": ROBOFLOW_MODEL_ID,
            "trained_images": 9665,
            "detected_buildings": building_count,
            "ai_raw_detections": len(roboflow_features),
            "osm_vector_buildings": len(osm_buildings),
            "total_features": len(final_features),
            "roboflow_error": roboflow_error,
            "center": {"lat": req.latitude, "lon": req.longitude},
            "zoom": req.zoom,
            "scan_zoom": scan_zoom,
            "bounds": bounds,
            "tiles": roboflow_res.get("tiles", 4),
        }
    })


# ── /api/roads (Overpass API Yol Ağları Endpoint'i) ──────────────────────────
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
    (highways, streets) OpenStreetMap Overpass API üzerinden çeker ve standart
    GeoJSON (LineString) FeatureCollection formatında döndürür.
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
    Overpass API üzerinden yol ağlarını (LineString GeoJSON) döndürür.
    """
    bounds = _resolve_road_bounds(
        req.south, req.west, req.north, req.east,
        req.latitude, req.longitude, req.zoom
    )
    road_geojson = await asyncio.to_thread(fetch_osm_roads_overpass, bounds, MAPBOX_ACCESS_TOKEN)
    return JSONResponse(content=road_geojson)


# ── /roboflow-status & /model-status ──────────────────────────────────────────
@app.get("/roboflow-status")
async def roboflow_status():
    """Roboflow Instance Segmentation model durumunu döndürür."""
    return {
        "status": "ready",
        "model_id": ROBOFLOW_MODEL_ID,
        "api_url": ROBOFLOW_API_URL,
        "api_key_configured": bool(ROBOFLOW_API_KEY),
        "trained_images": 9665,
        "task": "instance_segmentation",
        "class": "building",
        "default_height": 15.0,
    }


@app.get("/model-status")
async def model_status():
    """Sistem ve model hazırlık durumunu döndürür."""
    return {
        "mapbox_configured": bool(MAPBOX_ACCESS_TOKEN) and not MAPBOX_ACCESS_TOKEN.startswith("TODO"),
        "roboflow": {
            "ready": bool(ROBOFLOW_API_KEY),
            "model_id": ROBOFLOW_MODEL_ID,
            "api_url": ROBOFLOW_API_URL,
            "trained_images": 9665,
            "task": "instance_segmentation",
        },
    }


