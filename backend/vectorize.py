"""
vectorize.py -- GeoSim Hibrit Vektörleştirme Modülü (Coğrafi Koordinat Destekli)
=================================================================================
- Mapbox Static Images Bounding Box & Pixel-to-Geo İnterpolasyonu
- Roboflow AI 2x2 Tiling Bina Segmentasyonu & GeoJSON Dönüşümü
- Mapbox / OpenStreetMap Vektör Karoları (MVT) & Hibrit Bina Birleştirici
- Overpass API / OSM Hazır Yol Ağları (LineString GeoJSON) Çekici
"""

import os
import io
import copy
import base64
import time
import math
import uuid
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import requests
from requests.adapters import HTTPAdapter

from prediction_pipeline import deduplicate_predictions, preserve_polygon, PIPELINE_VERSION

logger = logging.getLogger("vectorize")

# Kalıcı HTTP bağlantı havuzu ve bellek içi önbellekler
_HTTP_SESSION = requests.Session()
_HTTP_ADAPTER = HTTPAdapter(pool_connections=16, pool_maxsize=16, max_retries=1)
_HTTP_SESSION.mount("https://", _HTTP_ADAPTER)
_HTTP_SESSION.mount("http://", _HTTP_ADAPTER)

_MVT_TILE_CACHE: dict[tuple[int, int, int], bytes] = {}
_VECTOR_BBOX_CACHE: dict[tuple[float, float, float, float], tuple[float, list[dict], list[dict], dict]] = {}
_VECTOR_CACHE_TTL_S = 600.0
_SHARED_POOL = ThreadPoolExecutor(max_workers=8)


def _get_http_session() -> requests.Session:
    return _HTTP_SESSION


def _load_env_file() -> None:
    """Backend dizinindeki .env dosyasından ortam değişkenlerini yükler."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        except Exception:
            pass


_load_env_file()

# Mapbox Static Images API varsayılan ana görüntü boyutu ve tile tabanı
MAIN_IMAGE_W     = 1280  # Ana piksel düzlemi genişliği (px)
MAIN_IMAGE_H     = 1280  # Ana piksel düzlemi yüksekliği (px)
MAPBOX_TILE_SIZE = 512   # Mapbox GL / Static API zoom tabanı: 512 px tile


# ── Koordinat Dönüşümü (Bounding Box & Pixel-to-Geo İnterpolasyonu) ──────────
def get_mapbox_bounding_box(
    center_lat: float,
    center_lon: float,
    zoom: float,
    img_w: int = MAIN_IMAGE_W,
    img_h: int = MAIN_IMAGE_H,
    pixel_ratio: int = 1,
) -> dict:
    """
    Mapbox Static API'den (center_lat, center_lon, zoom) ile çekilen
    1280x1280 görüntünün gerçek dünya Bounding Box (Kuzey, Güney, Doğu, Batı)
    sınır koordinatlarını hesaplar.

    Döndürür:
        {
            "north": float,  # Üst sınır enlemi (y = 0)
            "south": float,  # Alt sınır enlemi (y = img_h)
            "east":  float,  # Sağ sınır boylamı (x = img_w)
            "west":  float,  # Sol sınır boylamı (x = 0)
            "polygon": [[west, north], [east, north], [east, south], [west, south], [west, north]]
        }
    """
    world_size = float(MAPBOX_TILE_SIZE * (2.0 ** zoom) * pixel_ratio)

    # Merkez noktasının dünya piksel koordinatları (Web Mercator)
    cx = (center_lon + 180.0) / 360.0 * world_size
    sin_lat = math.sin(math.radians(center_lat))
    sin_lat = min(max(sin_lat, -0.9999), 0.9999)
    cy = (0.5 - math.log((1.0 + sin_lat) / (1.0 - sin_lat)) / (4.0 * math.pi)) * world_size

    half_w = float(img_w) / 2.0
    half_h = float(img_h) / 2.0

    # Batı (x = 0) ve Doğu (x = img_w) boylamları
    west = ((cx - half_w) / world_size) * 360.0 - 180.0
    east = ((cx + half_w) / world_size) * 360.0 - 180.0

    # Kuzey (y = 0, üst kenar) ve Güney (y = img_h, alt kenar) enlemleri
    n_north = math.pi - (2.0 * math.pi * (cy - half_h)) / world_size
    north = math.degrees(math.atan(math.sinh(n_north)))

    n_south = math.pi - (2.0 * math.pi * (cy + half_h)) / world_size
    south = math.degrees(math.atan(math.sinh(n_south)))

    return {
        "north": round(north, 7),
        "south": round(south, 7),
        "east":  round(east, 7),
        "west":  round(west, 7),
        "polygon": [
            [round(west, 7), round(north, 7)],
            [round(east, 7), round(north, 7)],
            [round(east, 7), round(south, 7)],
            [round(west, 7), round(south, 7)],
            [round(west, 7), round(north, 7)],
        ],
    }


def pixel_to_geo_bbox(
    px: float,
    py: float,
    bounds: dict,
    img_w: int = MAIN_IMAGE_W,
    img_h: int = MAIN_IMAGE_H,
) -> list[float]:
    """
    Ana resimdeki (1280x1280) piksel (px, py) noktasını, Mapbox Bounding Box
    (north, south, east, west) sınırları içinde matematiksel oranlamayla (interpolation)
    gerçek [longitude, latitude] koordinatına dönüştürür.
    """
    north = float(bounds["north"])
    south = float(bounds["south"])
    east  = float(bounds["east"])
    west  = float(bounds["west"])

    w = float(max(img_w, 1))
    h = float(max(img_h, 1))

    # Piksel oranları (0.0 .. 1.0)
    ratio_x = float(px) / w
    ratio_y = float(py) / h

    # Matematiksel interpolasyon (Y ekseni Kuzey -> Güney ters çalışır)
    lon = west + ratio_x * (east - west)
    yn = math.asinh(math.tan(math.radians(north)))
    ys = math.asinh(math.tan(math.radians(south)))
    lat = math.degrees(math.atan(math.sinh(yn + ratio_y * (ys - yn))))

    return [round(lon, 7), round(lat, 7)]


# ── Roboflow Ready Model Entegrasyonu ─────────────────────────────────────────
# 9.665 görselle eğitilmiş Instance Segmentation Modeli (.env'den okunur)
ROBOFLOW_API_URL  = os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com")
ROBOFLOW_API_KEY  = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "buildings-instance-segmentation/3")


def box_iou(b1: tuple, b2: tuple) -> float:
    """İki sınırlayıcı kutunun (x_center, y_center, w, h) IoU (kesişim/birleşim) oranını hesaplar."""
    x1_min, y1_min = b1[0] - b1[2] / 2, b1[1] - b1[3] / 2
    x1_max, y1_max = b1[0] + b1[2] / 2, b1[1] + b1[3] / 2
    x2_min, y2_min = b2[0] - b2[2] / 2, b2[1] - b2[3] / 2
    x2_max, y2_max = b2[0] + b2[2] / 2, b2[1] + b2[3] / 2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = b1[2] * b1[3]
    area2 = b2[2] * b2[3]
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0


def _tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    """Bir eksen boyunca örtüşmeli tile başlangıç noktalarını üretir (son tile kenara hizalanır)."""
    if length <= tile:
        return [0]
    stride = max(1, tile - overlap)
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] + tile < length:
        starts.append(length - tile)
    return starts


def _is_single_building_geom(
    g,
    scale1: float = 1.0,
    scale2: float = 1.0,
) -> tuple[bool, float, float, float]:
    """
    Poligonun tek bir binaya mı yoksa birden fazla binayı/avluyu yutan bir lekeye
    (multi-building blob) mi ait olduğunu geometri ve dikdörtgenlik analiziyle belirler.
    Döndürür: (is_single, rw, rh, rectangularity)
    """
    if g is None or g.is_empty or g.area <= 16.0:
        return False, 0.0, 0.0, 0.0
    if g.geom_type == "MultiPolygon":
        g = max(g.geoms, key=lambda q: q.area)
    coords = np.array(g.exterior.coords[:-1], dtype=np.float32)
    if len(coords) < 3:
        return False, 0.0, 0.0, 0.0
    (_, _), (rw, rh), _ = cv2.minAreaRect(coords.reshape((-1, 1, 2)))
    rw, rh = float(rw), float(rh)
    rect_area = max(rw * rh, 1.0)
    area = float(g.area)
    rectangularity = area / rect_area
    max_dim = max(rw, rh)
    min_dim = max(min(rw, rh), 1.0)

    # 1280x1280 referans ölçeğinde (zoom 17'de 1 px ≈ 0.44m):
    # Tek bir bina tipik olarak <= 3200 px² (~620 m²) ve kenarı <= 82 px (~36m) olur.
    if area > 3200.0 * scale2:
        return False, rw, rh, rectangularity
    if max_dim > 82.0 * scale1 or rect_area > 4200.0 * scale2:
        return False, rw, rh, rectangularity
    if max_dim / min_dim > 5.0:
        return False, rw, rh, rectangularity
    if area > 1200.0 * scale2 and rectangularity < 0.52:
        return False, rw, rh, rectangularity
    if rectangularity < 0.38:
        return False, rw, rh, rectangularity
    return True, rw, rh, rectangularity


def _merge_tile_predictions(
    preds: list[dict],
    img_w: int = MAIN_IMAGE_W,
    img_h: int = MAIN_IMAGE_H,
    return_blobs: bool = False,
):
    """
    Örtüşmeli karolardan gelen tespitleri tek listeye indirger:
    1. Çok binalı lekeleri (multi-building blobs) ayırır: aşırı büyük, düşük
       dikdörtgenlikli veya içinde birden fazla bina barındıran maskeleri tekil
       bina listesinden çıkarır (görüntü tabanlı watershed ile ayrıştırılmak üzere).
    2. Tekil binalar arasında poligon tabanlı NMS uygular; karo sınırında bölünmüş
       parçaları yalnızca birleşim hala tekil bina ölçülerindeyse birleştirir.
    """
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union
    from shapely.strtree import STRtree

    scale1 = max(float(max(img_w, img_h)) / 1280.0, 0.25)
    scale2 = scale1 * scale1

    geoms = []
    single_flags = []
    for p in preds:
        try:
            pts = p.get("points") or []
            g = Polygon([(float(q["x"]), float(q["y"])) for q in pts]).buffer(0).simplify(0.75)
            if isinstance(g, MultiPolygon):
                g = max(g.geoms, key=lambda q: q.area)
        except Exception:
            g = Polygon()
        geoms.append(g)
        is_single, _, _, _ = _is_single_building_geom(g, scale1=scale1, scale2=scale2)
        single_flags.append(is_single)

    # 1. Çok binalı leke (blob) tespiti
    tree = STRtree(geoms) if geoms else None
    blobs: set[int] = {i for i, ok in enumerate(single_flags) if not ok and not geoms[i].is_empty}

    if tree is not None:
        for i, g in enumerate(geoms):
            if g.is_empty or i in blobs:
                continue
            inside = []
            for j in tree.query(g):
                j = int(j)
                s = geoms[j]
                if j == i or s.is_empty or s.area >= 0.65 * g.area:
                    continue
                if g.intersects(s) and g.intersection(s).area > 0.62 * s.area:
                    inside.append(s)
            distinct = []
            for s in sorted(inside, key=lambda q: -q.area):
                if all(s.intersection(t).area < 0.45 * min(s.area, t.area) for t in distinct):
                    distinct.append(s)
            if len(distinct) >= 2:
                blobs.add(i)
            elif len(distinct) == 1 and distinct[0].area < 0.42 * g.area and g.area > 1600.0 * scale2:
                blobs.add(i)

    # 2. Tekil binalar için poligon tabanlı NMS + kontrollü karo dikişi birleştirme
    order = sorted(
        (i for i in range(len(preds)) if i not in blobs and not geoms[i].is_empty),
        key=lambda i: (preds[i].get("_cut", False), -float(preds[i].get("confidence", 0))),
    )
    kept: list[int] = []
    merged: set[int] = set()
    for i in order:
        p = preds[i]
        gi = geoms[i]
        if gi.is_empty:
            continue
        box_p = (p["x"], p["y"], p["width"], p["height"])
        target = None
        for k in kept:
            d = preds[k]
            if abs(box_p[0] - d["x"]) * 2.0 > (box_p[2] + d["width"]) or abs(box_p[1] - d["y"]) * 2.0 > (box_p[3] + d["height"]):
                continue
            gk = geoms[k]
            if gk.is_empty:
                continue
            if not gi.intersects(gk):
                continue
            inter = gi.intersection(gk).area if gi.intersects(gk) else 0.0
            if inter <= 0.0:
                continue
            min_a = max(min(gi.area, gk.area), 1e-6)
            union_a = max(gi.area + gk.area - inter, 1e-6)
            poly_iou = inter / union_a
            poly_iomin = inter / min_a

            if p.get("_cut") and d.get("_cut") and p.get("tile") != d.get("tile"):
                if poly_iomin >= 0.25:
                    u = unary_union([gk, gi]).buffer(0)
                    if isinstance(u, MultiPolygon):
                        u = max(u.geoms, key=lambda q: q.area)
                    ok_u, _, _, _ = _is_single_building_geom(u, scale1=scale1, scale2=scale2)
                    if ok_u:
                        geoms[k] = u
                        x0, y0, x1, y1 = u.bounds
                        d.update(x=(x0 + x1) / 2, y=(y0 + y1) / 2, width=x1 - x0, height=y1 - y0)
                        merged.add(k)
                        target = k
                        break
            if poly_iou > 0.35 or poly_iomin > 0.58:
                target = k
                break
        if target is None:
            kept.append(i)

    out = []
    for k in kept:
        p = dict(preds[k])
        if k in merged and isinstance(geoms[k], Polygon):
            p["points"] = [{"x": float(x), "y": float(y)} for x, y in geoms[k].exterior.coords[:-1]]
        p.pop("_cut", None)
        out.append(p)

    if return_blobs:
        blob_list = []
        for idx in sorted(blobs):
            bp = dict(preds[idx])
            bp.pop("_cut", None)
            blob_list.append((bp, geoms[idx]))
        return out, blob_list
    return out


def _split_and_recover_satellite_roofs(
    pil_img,
    single_preds: list[dict],
    blob_items: list[tuple[dict, object]],
    confidence_threshold: float = 0.25,
) -> list[dict]:
    """
    1. Roboflow'un birden fazla binayı tek leke (blob) olarak birleştirdiği maskeleri,
       uydu görüntüsündeki gerçek çatı renkleri, bina arası gölge koridorları, ağaçlar
       ve Sobel kenar gradyanlarını kullanarak (Distance Transform + Watershed) tekil
       binalara ayrıştırır.
    2. Modelin atladığı belirgin kiremit/terrakota çatıları uydu görüntüsünden çıkarır.
    """
    img_w, img_h = pil_img.size
    if img_w < 256 or img_h < 256:
        return single_preds

    work_w = min(img_w, 1280)
    work_h = min(img_h, 1280)
    sx = work_w / float(img_w)
    sy = work_h / float(img_h)
    inv_sx = float(img_w) / work_w
    inv_sy = float(img_h) / work_h

    work_img = pil_img.resize((work_w, work_h), resample=2) if (work_w != img_w or work_h != img_h) else pil_img
    rgb = np.asarray(work_img, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return single_preds

    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)

    # Parlaklık (Luminance) ve renk farkları
    lum = ((77 * r + 150 * g + 29 * b) >> 8).astype(np.uint8)
    warm_rg = r - g
    warm_rb = r - b

    # Eğer görüntü tamamen düz/yapay test görüntüsüyse (varyans yoksa) çık
    if float(lum.std()) < 4.0:
        return single_preds

    # 1. Çatı olmayan alanlar: Ağaç/bitki örtüsü, koyu bina gölgeleri, gri asfalt/zemin
    veg_mask = ((g > r + 3) & (g > b)) | ((2 * g - r - b) > 14)
    shadow_mask = lum < 50
    grey_pavement = (np.abs(warm_rg) <= 8) & (np.abs(warm_rb) <= 12) & (lum >= 50) & (lum <= 172)

    # 2. Kiremit / Terrakota çatı maskesi (Güneşli + gölgeli kırma çatı yüzeyleri ve kahverengi kiremitler)
    terracotta_sun = (warm_rg >= 11) & (warm_rb >= 15) & (r >= 66) & (g <= 210) & (~veg_mask) & (~shadow_mask)
    terracotta_shade = (
        (warm_rg >= 7)
        & (warm_rb >= 11)
        & (r >= 52)
        & (g <= 178)
        & (r * 100 >= g * 111)
        & (r * 100 >= b * 117)
        & (~veg_mask)
        & (~shadow_mask)
    )
    terracotta_loose = terracotta_sun | terracotta_shade
    terracotta_strong = (
        ((warm_rg >= 12) & (warm_rb >= 16) & (r >= 66) & (g <= 205))
        | ((warm_rg >= 7) & (warm_rb >= 11) & (r >= 52) & (g <= 175) & (r * 100 >= g * 111) & (r * 100 >= b * 117))
    ) & (~veg_mask) & (~shadow_mask)

    # 3. Çatı sınırlarını birbirinden ayıran Sobel gradyan haritası
    # Not: Tekil kırma çatıların iç mahya çizgilerini (iki tarafı da kiremit olan orta gradyanları)
    # koruyup, çatı dış kenarlarını ve bitişik nizam bina ayrımlarını bariyer olarak kullan
    lum_blur = cv2.GaussianBlur(lum, (3, 3), 0)
    rg_clamped = np.clip(warm_rg + 128, 0, 255).astype(np.uint8)
    gx_l = cv2.Sobel(lum_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy_l = cv2.Sobel(lum_blur, cv2.CV_32F, 0, 1, ksize=3)
    gx_r = cv2.Sobel(rg_clamped, cv2.CV_32F, 1, 0, ksize=3)
    gy_r = cv2.Sobel(rg_clamped, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx_l, gy_l) + 0.75 * np.hypot(gx_r, gy_r)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    deep_roof_interior = cv2.erode(terracotta_loose.astype(np.uint8) * 255, k5, iterations=1) > 0
    grad_p80 = max(float(np.percentile(grad, 80)), 38.0)
    grad_p89 = max(float(np.percentile(grad, 89)), 54.0)
    edge_barrier = (grad > grad_p89) | ((grad > grad_p80) & (~deep_roof_interior))
    split_barrier = grad > max(float(np.percentile(grad, 76)), 34.0)

    # 4. Mevcut tekil bina tahminlerini maskele (tekrar üretmemek için)
    existing_mask = np.zeros((work_h, work_w), dtype=np.uint8)
    for sp in single_preds:
        pts = sp.get("points") or []
        if len(pts) >= 3:
            arr = np.array([[int(round(float(q["x"]) * sx)), int(round(float(q["y"]) * sy))] for q in pts], dtype=np.int32)
            cv2.fillPoly(existing_mask, [arr], 255)
    if np.any(existing_mask):
        existing_mask = cv2.dilate(existing_mask, k5, iterations=1)

    # 5. Çok binalı Roboflow lekelerinin (blob) içindeki çatı yüzeyleri
    blob_mask = np.zeros((work_h, work_w), dtype=np.uint8)
    blob_conf_map = np.zeros((work_h, work_w), dtype=np.float32)
    for bp, _ in blob_items:
        pts = bp.get("points") or []
        if len(pts) >= 3:
            arr = np.array([[int(round(float(q["x"]) * sx)), int(round(float(q["y"]) * sy))] for q in pts], dtype=np.int32)
            c_val = float(bp.get("confidence", 0.55))
            tmp = np.zeros((work_h, work_w), dtype=np.uint8)
            cv2.fillPoly(tmp, [arr], 255)
            blob_mask |= tmp
            blob_conf_map = np.maximum(blob_conf_map, (tmp > 0).astype(np.float32) * c_val)

    # Blob içindeki gerçek çatı pikselleri: bitki, gölge, gri sokak ve dış çatı sınırlarını çıkar
    bright_concrete_roof = (lum > 176) & (~veg_mask)
    blob_roofs = (
        (blob_mask > 0)
        & (~veg_mask)
        & (~shadow_mask)
        & (terracotta_loose | bright_concrete_roof | (~grey_pavement))
        & (~edge_barrier)
    )

    # Tüm görüntüdeki kiremit çatılar (mahya çizgilerini koruyup dış sınırları ayır)
    global_terracotta = terracotta_strong & (~edge_barrier)
    global_terracotta = cv2.morphologyEx(global_terracotta.astype(np.uint8) * 255, cv2.MORPH_OPEN, k3) > 0

    candidate = ((blob_roofs | global_terracotta) & (existing_mask == 0)).astype(np.uint8) * 255
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, k3, iterations=1)
    candidate_closed = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, k3, iterations=1)
    candidate_closed[veg_mask | shadow_mask | (existing_mask > 0)] = 0

    # Çatı içindeki küçük baca / mahya / ışıklık deliklerini (< 75 work_px²) doldur
    inv_cand = (candidate_closed == 0).astype(np.uint8)
    n_holes, hole_lbls, hole_stats, _ = cv2.connectedComponentsWithStats(inv_cand, connectivity=4)
    for h_id in range(1, n_holes):
        if int(hole_stats[h_id, cv2.CC_STAT_AREA]) < 75:
            hx = int(hole_stats[h_id, cv2.CC_STAT_LEFT])
            hy = int(hole_stats[h_id, cv2.CC_STAT_TOP])
            hw = int(hole_stats[h_id, cv2.CC_STAT_WIDTH])
            hh = int(hole_stats[h_id, cv2.CC_STAT_HEIGHT])
            sub_h = hole_lbls[hy:hy + hh, hx:hx + hw]
            candidate_closed[hy:hy + hh, hx:hx + hw][sub_h == h_id] = 255

    grad_u8 = np.clip(grad * (255.0 / max(float(np.percentile(grad, 98)), 1.0)), 0, 255).astype(np.uint8)
    peak_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    # 6. Bağlı bileşenleri dar kutularında (cropped bbox) analiz et:
    # - Tekil bina boyutundaki bileşenleri (95..1150 work_px²) doğrudan koru
    # - Çoklu bina kümelerini yerel Watershed + dik eksen kesimiyle tekil binalara ayır
    def _extract_sub_masks(comp_u8: np.ndarray, ox: int, oy: int, depth: int = 0) -> list[tuple[np.ndarray, int, int]]:
        ys, xs = np.nonzero(comp_u8)
        area = len(xs)
        if area < 95:
            return []
        pts = np.stack([xs, ys], axis=-1).astype(np.float32)
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(pts.reshape((-1, 1, 2)))
        max_d = max(rw, rh)
        min_d = max(min(rw, rh), 1.0)
        rect_a = max(rw * rh, 1.0)
        rect_ratio = area / rect_a

        needs_split = (
            area > 1150
            or max_d > 52.0
            or (area > 580 and rect_ratio < 0.42)
            or (area > 460 and (max_d / min_d) > 3.5)
        )

        # Adım A: İlk derinlikte çoklu bina kümesini yerel Watershed ile doğal çatı sınırlarından böl
        if depth == 0 and needs_split:
            h_c, w_c = comp_u8.shape
            local_sb = split_barrier[oy:oy + h_c, ox:ox + w_c]
            local_grad = grad_u8[oy:oy + h_c, ox:ox + w_c]
            core = comp_u8.copy()
            core[local_sb] = 0
            dist_local = cv2.distanceTransform(core, cv2.DIST_L2, 5)
            lmax = (dist_local == cv2.dilate(dist_local, peak_kernel)) & (dist_local >= 3.0)
            pk_u8 = cv2.dilate(lmax.astype(np.uint8) * 255, k3, iterations=1)
            pk_u8[comp_u8 == 0] = 0
            n_mk, mk = cv2.connectedComponents(pk_u8)
            if n_mk > 2:
                ws_in = cv2.cvtColor(local_grad, cv2.COLOR_GRAY2BGR)
                ws_mk = mk.astype(np.int32)
                ws_mk[comp_u8 == 0] = -1
                ws_mk[(comp_u8 > 0) & (pk_u8 == 0)] = 0
                cv2.watershed(ws_in, ws_mk)
                ws_fg_local = (ws_mk > 0).astype(np.uint8) * 255
                n_wcc, wcc_lbl, wcc_st, _ = cv2.connectedComponentsWithStats(ws_fg_local, connectivity=4)
                ws_res: list[tuple[np.ndarray, int, int]] = []
                for wid in range(1, n_wcc):
                    if int(wcc_st[wid, cv2.CC_STAT_AREA]) < 95:
                        continue
                    wx = int(wcc_st[wid, cv2.CC_STAT_LEFT])
                    wy = int(wcc_st[wid, cv2.CC_STAT_TOP])
                    ww = int(wcc_st[wid, cv2.CC_STAT_WIDTH])
                    wh = int(wcc_st[wid, cv2.CC_STAT_HEIGHT])
                    piece_crop = ((wcc_lbl[wy:wy + wh, wx:wx + ww] == wid).astype(np.uint8)) * 255
                    ws_res.extend(_extract_sub_masks(piece_crop, ox + wx, oy + wy, depth + 1))
                if len(ws_res) >= 2:
                    return ws_res

        # Adım B: Hala büyük veya çoklu bina boyutundaysa ana eksene dik çizgiyle ikiye ayır
        max_split_depth = 3 if (area > 1600 or max_d > 60.0) else 2
        if depth < max_split_depth and needs_split:
            rad = math.radians(ang)
            if rw < rh:
                rad += math.pi / 2.0
            nx, ny = -math.sin(rad), math.cos(rad)
            cut = comp_u8.copy()
            x0, y0 = int(round(cx - nx * max_d)), int(round(cy - ny * max_d))
            x1, y1 = int(round(cx + nx * max_d)), int(round(cy + ny * max_d))
            cv2.line(cut, (x0, y0), (x1, y1), 0, thickness=2)
            n_sub, sub_lbl, sub_st, _ = cv2.connectedComponentsWithStats(cut, connectivity=4)
            if n_sub > 2:
                res: list[tuple[np.ndarray, int, int]] = []
                for sid in range(1, n_sub):
                    if int(sub_st[sid, cv2.CC_STAT_AREA]) < 95:
                        continue
                    sx0 = int(sub_st[sid, cv2.CC_STAT_LEFT])
                    sy0 = int(sub_st[sid, cv2.CC_STAT_TOP])
                    sw0 = int(sub_st[sid, cv2.CC_STAT_WIDTH])
                    sh0 = int(sub_st[sid, cv2.CC_STAT_HEIGHT])
                    sub_crop = ((sub_lbl[sy0:sy0 + sh0, sx0:sx0 + sw0] == sid).astype(np.uint8)) * 255
                    res.extend(_extract_sub_masks(sub_crop, ox + sx0, oy + sy0, depth + 1))
                if res:
                    return res

        if area > 2600 or max_d > 74.0 or (max_d / min_d > 4.2):
            return []
        if rect_ratio < 0.42:
            return []
        return [(comp_u8, ox, oy)]

    recovered: list[dict] = []
    num_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(candidate_closed, connectivity=4)

    for lbl in range(1, num_cc):
        off_x0 = int(cc_stats[lbl, cv2.CC_STAT_LEFT])
        off_y0 = int(cc_stats[lbl, cv2.CC_STAT_TOP])
        bw_c = int(cc_stats[lbl, cv2.CC_STAT_WIDTH])
        bh_c = int(cc_stats[lbl, cv2.CC_STAT_HEIGHT])
        px_count = int(cc_stats[lbl, cv2.CC_STAT_AREA])
        if px_count < 95 or bw_c < 6 or bh_c < 6:
            continue

        sub_cc = cc_labels[off_y0:off_y0 + bh_c, off_x0:off_x0 + bw_c]
        comp_local = ((sub_cc == lbl).astype(np.uint8)) * 255

        for piece, pox, poy in _extract_sub_masks(comp_local, off_x0, off_y0):
            cnts, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)
            c_area = float(cv2.contourArea(cnt))
            if c_area < 85.0 or c_area > 2600.0:
                continue

            ph, pw = piece.shape
            local_blob_c = float(blob_conf_map[poy:poy + ph, pox:pox + pw][piece > 0].mean()) if np.any(piece > 0) else 0.0
            # Roboflow blob'u dışındaki saf renk tabanlı adaylarda küçük kaldırım/avlu lekelerini ele
            if local_blob_c <= 0.12 and c_area < 110.0:
                continue
            # Mevcut bir tekil binanın saçak/kenar taşmasıysa tekrar kutu üretme
            if np.any(existing_mask[poy:poy + ph, pox:pox + pw]):
                rect_check = cv2.minAreaRect(cnt)
                box_check = np.int32(cv2.boxPoints(rect_check))
                box_m = np.zeros((ph, pw), dtype=np.uint8)
                cv2.fillPoly(box_m, [box_check], 255)
                box_px = max(int(np.count_nonzero(box_m)), 1)
                exist_ov = int(np.count_nonzero((box_m > 0) & (existing_mask[poy:poy + ph, pox:pox + pw] > 0)))
                if exist_ov / float(box_px) > 0.22:
                    continue

            # Orijinal görüntü koordinatlarına ölçekle
            pts_global = []
            xs_g, ys_g = [], []
            for pt in cnt.reshape(-1, 2):
                gx = min(max((float(pt[0]) + pox) * inv_sx, 0.0), float(img_w))
                gy = min(max((float(pt[1]) + poy) * inv_sy, 0.0), float(img_h))
                pts_global.append({"x": gx, "y": gy})
                xs_g.append(gx)
                ys_g.append(gy)
            if len(pts_global) < 3:
                continue

            # Güven skoru: blob içindeyse blob güvenini, kiremit çatıysa renk/kontrast gücünü baz al
            local_rg = float(warm_rg[poy:poy + ph, pox:pox + pw][piece > 0].mean()) if np.any(piece > 0) else 0.0
            color_conf = min(0.84, max(0.42, 0.42 + (local_rg - 15.0) * 0.012))
            conf = round(max(local_blob_c * 0.92, color_conf), 4)
            if conf < confidence_threshold:
                continue

            min_gx, max_gx = min(xs_g), max(xs_g)
            min_gy, max_gy = min(ys_g), max(ys_g)
            w_g = max(max_gx - min_gx, 1.0)
            h_g = max(max_gy - min_gy, 1.0)

            recovered.append({
                "x": round((min_gx + max_gx) / 2.0, 2),
                "y": round((min_gy + max_gy) / 2.0, 2),
                "width": round(w_g, 2),
                "height": round(h_g, 2),
                "confidence": conf,
                "class": "Building",
                "class_id": 1,
                "prediction_source": "color_heuristic",
                "detection_id": f"rf_split_{ lbl }_{ len(recovered) + 1 }",
                "points": pts_global,
                "mask_format": "polygon",
                "offset_x": 0,
                "offset_y": 0,
                "offset_applied": True,
            })

    if not recovered:
        return single_preds

    # Son aşama: Tekil Roboflow binaları ile ayrıştırılan/kurtarılan çatıları birleştirip son NMS'ten geçir
    combined = list(single_preds) + recovered
    return _merge_tile_predictions(combined, img_w=img_w, img_h=img_h, return_blobs=False)



def scan_satellite_image_roboflow(
    image,
    api_url: str = ROBOFLOW_API_URL,
    api_key: str = ROBOFLOW_API_KEY,
    model_id: str = ROBOFLOW_MODEL_ID,
    enable_tiling: bool = True,
    confidence_threshold: float = 0.25,
    tile_size: int | None = None,
    overlap: int = 0,
    max_workers: int = 4,
    target_size: tuple[int, int] = (MAIN_IMAGE_W, MAIN_IMAGE_H),
    processing_profile: str = "conservative",
) -> dict:
    """
    Roboflow Instance Segmentation (2x2 Tiling & Offset Düzeltmeli):
    1. Ana görüntüyü 1280x1280 piksel ana düzleme oturtur.
    2. Görüntüyü 4 eşit çeyrek parçaya (2x2 Grid: 640x640) böler:
         - Sol-Üst (Top-Left)     : offset_x = 0,   offset_y = 0
         - Sağ-Üst (Top-Right)    : offset_x = 640, offset_y = 0
         - Sol-Alt (Bottom-Left)  : offset_x = 0,   offset_y = 640
         - Sağ-Alt (Bottom-Right) : offset_x = 640, offset_y = 640
    3. Her çeyrek resmin model sonuçlarındaki piksel [x, y] noktalarına ilgili
       offset_x ve offset_y değerlerini ekleyerek tüm binaları 1280x1280'lik
       ana piksel düzleminde gerçek yerlerine oturtur.
    """
    from PIL import Image

    # Görseli PIL Image nesnesine dönüştür
    if isinstance(image, str):
        pil_img = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    elif isinstance(image, Image.Image):
        pil_img = image.convert("RGB")
    else:
        raise ValueError("Desteklenmeyen görsel formatı!")

    # Ana görüntüyü 1280x1280 ana piksel düzlemine sabitle
    if target_size and pil_img.size != target_size:
        pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)

    img_w, img_h = pil_img.size
    total_area = img_w * img_h

    def _infer_single_pil(tile_pil, req_timeout=(1.8, 2.5)) -> dict | None:
        """
        Görseli bellekte hızlı JPEG (quality=80) olarak sıkıştırıp kalıcı HTTP
        bağlantı havuzu üzerinden doğrudan Roboflow Serverless REST API'ye gönderir.
        """
        buf = io.BytesIO()
        tile_pil.save(buf, format="JPEG", quality=80)
        b64_payload = base64.b64encode(buf.getvalue()).decode("ascii")
        endpoint = f"{api_url.rstrip('/')}/{model_id.lstrip('/')}"
        resp = _get_http_session().post(
            endpoint,
            params={
                "api_key": api_key,
                "confidence": confidence_threshold,
            },
            data=b64_payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=req_timeout,
        )
        if resp.status_code == 200:
            data = resp.json()
            if not isinstance(data, dict) or not isinstance(data.get("predictions"), list):
                raise RuntimeError("Roboflow response has no valid predictions list")
            return data
        raise RuntimeError(f"Roboflow HTTP {resp.status_code}")

    if not enable_tiling:
        logger.info(f"Roboflow tek parça çıkarım yapılıyor: {model_id}")
        res = _infer_single_pil(pil_img, req_timeout=(3.0, 12.0)) or {}
        preds = res.get("predictions", [])
        res_w = float((res.get("image") or {}).get("width") or img_w)
        res_h = float((res.get("image") or {}).get("height") or img_h)
        sx, sy = img_w / max(res_w, 1.0), img_h / max(res_h, 1.0)
        normalized_preds = []
        for p in preds:
            if float(p.get("confidence", 0)) < confidence_threshold:
                continue
            pts = [
                {
                    "x": min(max(float(pt.get("x", 0)) * sx, 0.0), float(img_w)),
                    "y": min(max(float(pt.get("y", 0)) * sy, 0.0), float(img_h)),
                }
                for pt in p.get("points", [])
            ]
            np_pred = dict(p)
            np_pred["points"] = pts
            np_pred["x"] = float(p.get("x", 0)) * sx
            np_pred["y"] = float(p.get("y", 0)) * sy
            np_pred["width"] = float(p.get("width", 0)) * sx
            np_pred["height"] = float(p.get("height", 0)) * sy
            np_pred["offset_x"] = 0
            np_pred["offset_y"] = 0
            np_pred["offset_applied"] = True
            normalized_preds.append(np_pred)
        return {
            "predictions": normalized_preds,
            "raw_predictions": copy.deepcopy(normalized_preds),
            "processing_profile": processing_profile,
            "pipeline_version": PIPELINE_VERSION,
            "provider": "roboflow",
            "model_id": model_id,
            "image": {"width": img_w, "height": img_h},
            "tiles": 1,
            "failed_tiles": 0,
            "raw_tiles": [{"crop_box": [0, 0, img_w, img_h], "response": res}],
        }

    # ── 1. Görüntüyü 4 Parçaya (2x2 Çeyrek Grid) Böl ──────────────────────────
    mid_x = img_w // 2  # 1280 -> 640
    mid_y = img_h // 2  # 1280 -> 640

    if tile_size is None:
        # 2x2 Grid: (crop_box, offset_x, offset_y)
        quadrants = [
            ((0,     0,     mid_x, mid_y), 0,     0),      # Sol-Üst (Top-Left)
            ((mid_x, 0,     img_w, mid_y), mid_x, 0),      # Sağ-Üst (Top-Right)
            ((0,     mid_y, mid_x, img_h), 0,     mid_y),  # Sol-Alt (Bottom-Left)
            ((mid_x, mid_y, img_w, img_h), mid_x, mid_y),  # Sağ-Alt (Bottom-Right)
        ]
    else:
        quadrants = [
            ((x, y, min(x + tile_size, img_w), min(y + tile_size, img_h)), x, y)
            for y in _tile_starts(img_h, tile_size, overlap)
            for x in _tile_starts(img_w, tile_size, overlap)
        ]

    logger.info(f"Roboflow karo taraması başlatılıyor ({img_w}x{img_h} -> {len(quadrants)} parça)...")

    tile_timings: list[float] = []

    # Roboflow Serverless tek karo için ~2-4 s sürebiliyor (soğuk başlangıçta daha fazla);
    # zaman aşımı kısa tutulursa karolar düşer ve tespitlerin çoğu kaybolur.
    TILE_TIMEOUT = (3.0, 12.0)
    TILE_ATTEMPTS = 2

    def _infer(item):
        crop_box, offset_x, offset_y = item
        t0 = time.perf_counter()
        tile_img = pil_img.crop(crop_box)
        try:
            for attempt in range(1, TILE_ATTEMPTS + 1):
                try:
                    res = _infer_single_pil(tile_img, req_timeout=TILE_TIMEOUT)
                    return crop_box, offset_x, offset_y, res
                except Exception as e:
                    logger.warning(f"Tile {crop_box} çıkarım hatası (deneme {attempt}/{TILE_ATTEMPTS}): {type(e).__name__}")
            return crop_box, offset_x, offset_y, None
        finally:
            tile_timings.append(round(time.perf_counter() - t0, 3))

    tile_results = list(_SHARED_POOL.map(_infer, quadrants))

    failed_tiles = sum(1 for _, _, _, r in tile_results if r is None)
    all_predictions = []
    EDGE = 3  # px: tile kenarına yakın tespit kontrolü

    for (bx0, by0, bx1, by1), offset_x, offset_y, res in tile_results:
        if not res:
            continue
        preds = res.get("predictions", [])
        if not preds:
            continue

        tw = float(bx1 - bx0)
        th = float(by1 - by0)
        res_img_info = res.get("image") or {}
        res_w = float(res_img_info.get("width") or tw)
        res_h = float(res_img_info.get("height") or th)
        scale_x = tw / max(res_w, 1.0)
        scale_y = th / max(res_h, 1.0)

        for p in preds:
            conf = float(p.get("confidence", 0))
            if conf < confidence_threshold:
                continue

            w = float(p.get("width", 0)) * scale_x
            h = float(p.get("height", 0)) * scale_y
            poly_area = w * h

            # ── Hatalı / Yol Çizimlerini Filtrele ─────────────────────────────
            # a) Alan kontrolü: Poligonun alanı toplam görüntünün %15'inden büyükse ele
            if processing_profile == "legacy" and poly_area > total_area * 0.15:
                continue

            # b) En/Boy oranı kontrolü: 1:6'dan daha ince uzunsa (yol/cadde olma ihtimali) ele
            min_dim = max(min(w, h), 1.0)
            max_dim = max(w, h)
            if processing_profile == "legacy" and (max_dim / min_dim) > 6.0:
                continue

            # ── Tiling Offset (Kaydırma) Düzeltmesi ───────────────────────────
            shifted_points = []
            for pt in p.get("points", []):
                if isinstance(pt, dict):
                    local_x = float(pt.get("x", 0)) * scale_x
                    local_y = float(pt.get("y", 0)) * scale_y
                elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    local_x = float(pt[0]) * scale_x
                    local_y = float(pt[1]) * scale_y
                else:
                    continue

                global_x = min(max(local_x + offset_x, 0.0), float(img_w))
                global_y = min(max(local_y + offset_y, 0.0), float(img_h))
                shifted_points.append({"x": global_x, "y": global_y})

            if len(shifted_points) < 3:
                continue

            local_cx = float(p.get("x", 0)) * scale_x
            local_cy = float(p.get("y", 0)) * scale_y

            shifted_pred = dict(p)
            shifted_pred["points"] = shifted_points
            shifted_pred["x"] = local_cx + offset_x
            shifted_pred["y"] = local_cy + offset_y
            shifted_pred["width"] = w
            shifted_pred["height"] = h
            shifted_pred["offset_x"] = offset_x
            shifted_pred["offset_y"] = offset_y
            shifted_pred["offset_applied"] = True
            shifted_pred["tile"] = [bx0, by0, bx1, by1]

            lx0, ly0 = local_cx - w / 2.0, local_cy - h / 2.0
            lx1, ly1 = lx0 + w, ly0 + h
            shifted_pred["_cut"] = (
                (lx0 <= EDGE and bx0 > 0) or (ly0 <= EDGE and by0 > 0) or
                (lx1 >= tw - EDGE and bx1 < img_w) or (ly1 >= th - EDGE and by1 < img_h)
            )
            all_predictions.append(shifted_pred)

    # ── Leke Ayıklama, Çakışma Önleme (NMS), Karo Dikişi ve Uydu Çatı Ayrıştırma ─
    if processing_profile == "legacy":
        single_predictions, blob_items = _merge_tile_predictions(
            all_predictions, img_w=img_w, img_h=img_h, return_blobs=True)
        deduped_predictions = _split_and_recover_satellite_roofs(
            pil_img, single_predictions, blob_items, confidence_threshold)
    else:
        deduped_predictions = (all_predictions if processing_profile == "raw"
                               else deduplicate_predictions(all_predictions))

    logger.info(f"Roboflow karo taraması tamamlandı: {len(deduped_predictions)} bina {img_w}x{img_h} düzleme oturtuldu.")
    return {
        "predictions": deduped_predictions,
        "raw_predictions": all_predictions,
        "processing_profile": processing_profile,
        "pipeline_version": PIPELINE_VERSION,
        "provider": "roboflow",
        "model_id": model_id,
        "image": {"width": img_w, "height": img_h},
        "tiles": len(quadrants),
        "failed_tiles": failed_tiles,
        "tile_timings_s": sorted(tile_timings),
        "raw_tiles": [{"crop_box": list(crop), "response": response}
                      for crop, _, _, response in tile_results],
    }


def estimate_building_height_roboflow(
    area_px: float,
    img_area: float,
    default_height: float = 15.0
) -> float:
    """
    Bina yüksekliği tahmini (metre):
    - Varsayılan yükseklik: 15.0 m (~4-5 kat)
    - Geniş çatı alanına sahip yapılar için kat artışı ile 15m - 50m arası dinamik yükseklik.
    """
    if area_px <= 0 or img_area <= 0:
        return default_height
    ratio = area_px / img_area
    scale_bonus = 35.0 * (math.log1p(ratio * 300) / math.log1p(10))
    return round(max(default_height, default_height + scale_bonus), 1)


def _build_road_corridor_union(osm_roads: list[dict] | None, bounds: dict):
    """Yol çizgilerinden bina taşmasını önleyen tamponlu koridor geometrisi üretir."""
    if not osm_roads:
        return None
    from shapely.geometry import shape
    from shapely.ops import unary_union

    lat_mid = (float(bounds.get("north", 41.0)) + float(bounds.get("south", 41.0))) / 2.0
    deg_per_m = 1.0 / (111_320.0 * max(math.cos(math.radians(lat_mid)), 0.2))
    road_buffers = []
    for rf in osm_roads:
        try:
            rg = shape(rf["geometry"])
            if rg.is_empty or not rg.is_valid:
                continue
            w_m = float((rf.get("properties") or {}).get("width") or 3.0)
            half_w_deg = max(1.0, min(w_m * 0.35, 2.1)) * deg_per_m
            road_buffers.append(rg.buffer(half_w_deg, cap_style=2, join_style=2))
        except Exception:
            continue
    if not road_buffers:
        return None
    try:
        return unary_union(road_buffers)
    except Exception:
        return None


def roboflow_predictions_to_geojson(
    predictions_or_result,
    img_w: int = MAIN_IMAGE_W,
    img_h: int = MAIN_IMAGE_H,
    center_lat: float = 41.0082,
    center_lon: float = 28.9784,
    zoom: float = 16.5,
    bounds: dict | None = None,
    default_height: float = 15.0,
    simplify_epsilon: float = 1.5,
    osm_roads: list[dict] | None = None,
    processing_profile: str | None = None,
) -> dict:
    """
    Roboflow Instance Segmentation Poligon Dönüşümü ve deck.gl GeoJSON Çıktısı.
    """
    from shapely.geometry import Polygon, MultiPolygon

    if isinstance(predictions_or_result, dict):
        predictions = predictions_or_result.get("predictions", [])
        res_img = predictions_or_result.get("image") or {}
        img_w = int(res_img.get("width") or img_w or MAIN_IMAGE_W)
        img_h = int(res_img.get("height") or img_h or MAIN_IMAGE_H)
    else:
        predictions = predictions_or_result or []

    # ── Bounding Box (Kuzey, Güney, Doğu, Batı) Sınır Koordinatlarını Al ──────
    if bounds is None:
        bounds = get_mapbox_bounding_box(
            center_lat=center_lat,
            center_lon=center_lon,
            zoom=zoom,
            img_w=img_w,
            img_h=img_h,
        )

    result_meta = predictions_or_result if isinstance(predictions_or_result, dict) else {}
    processing_profile = processing_profile or result_meta.get("processing_profile", "conservative")
    provider = result_meta.get("provider", "roboflow")
    road_corridor = _build_road_corridor_union(osm_roads, bounds) if processing_profile == "legacy" else None
    features = []
    img_area = max(img_w * img_h, 1)

    for idx, pred in enumerate(predictions):
        points = pred.get("points", [])
        if not points or len(points) < 3:
            continue

        # ── 1. Tiling Offset Kontrolü (1280x1280 Ana Düzlem) ─────────────────
        offset_applied = bool(pred.get("offset_applied", False))
        off_x = 0.0 if offset_applied else float(pred.get("offset_x", 0.0))
        off_y = 0.0 if offset_applied else float(pred.get("offset_y", 0.0))

        raw_pts = []
        for p in points:
            if isinstance(p, dict):
                px = float(p.get("x", 0.0)) + off_x
                py = float(p.get("y", 0.0)) + off_y
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                px = float(p[0]) + off_x
                py = float(p[1]) + off_y
            else:
                continue

            px = min(max(px, 0.0), float(img_w))
            py = min(max(py, 0.0), float(img_h))
            raw_pts.append([px, py])

        if len(raw_pts) < 3:
            continue

        # Douglas-Peucker / Mimari Dikdörtgenleştirme (Orthogonalization)
        reg_pts = (_legacy_regularize_pixel_polygon(raw_pts, simplify_epsilon) if processing_profile == "legacy"
                   else regularize_pixel_polygon(raw_pts, 0 if processing_profile == "raw" else simplify_epsilon))
        if len(reg_pts) < 3:
            continue

        cnt_arr = np.array(reg_pts, dtype=np.float32)
        area_px = float(cv2.contourArea(cnt_arr))
        if area_px <= 16.0:
            w = float(pred.get("width", 0))
            h = float(pred.get("height", 0))
            area_px = max(w * h, 1.0)

        # ── 2. Pixel-to-Geo (Bounding Box İnterpolasyonu) ─────────────────────
        ring = []
        for pt in reg_pts:
            coord = pixel_to_geo_bbox(
                px=float(pt[0]),
                py=float(pt[1]),
                bounds=bounds,
                img_w=img_w,
                img_h=img_h,
            )
            if not ring or ring[-1] != coord:
                ring.append(coord)

        # ── 3. deck.gl GeoJSON Polygon Formatı [[[lon, lat], [lon, lat], ...]] ─
        if ring and ring[0] != ring[-1]:
            ring.append([ring[0][0], ring[0][1]])

        if len(ring) < 4:
            continue

        try:
            poly_geo = Polygon(ring).buffer(0)
            if isinstance(poly_geo, MultiPolygon):
                poly_geo = max(poly_geo.geoms, key=lambda q: q.area)
            if poly_geo.is_empty or not poly_geo.is_valid:
                continue
            if road_corridor is not None and poly_geo.intersects(road_corridor):
                r_inter = poly_geo.intersection(road_corridor).area
                if r_inter > 0.42 * poly_geo.area:
                    continue
                if r_inter > 0.08 * poly_geo.area:
                    clipped_g = poly_geo.difference(road_corridor).buffer(0)
                    if isinstance(clipped_g, MultiPolygon):
                        clipped_g = max(clipped_g.geoms, key=lambda q: q.area)
                    if isinstance(clipped_g, Polygon) and not clipped_g.is_empty and clipped_g.area >= 0.52 * poly_geo.area:
                        mrr = clipped_g.minimum_rotated_rectangle
                        if (
                            isinstance(mrr, Polygon)
                            and not mrr.is_empty
                            and (not road_corridor.intersects(mrr) or mrr.intersection(road_corridor).area <= 0.22 * mrr.area)
                        ):
                            poly_geo = mrr
                        elif len(clipped_g.exterior.coords) >= 5:
                            poly_geo = clipped_g
            ring = [[round(float(x), 7), round(float(y), 7)] for x, y in poly_geo.exterior.coords]
            if ring and ring[0] != ring[-1]:
                ring.append([ring[0][0], ring[0][1]])
            if len(ring) < 4:  # En az 4 köşe + kapanış noktası = 5 (üçgenleri engelle)
                continue
        except Exception:
            continue

        b_height = estimate_building_height_roboflow(area_px, img_area, default_height=default_height)
        floors = max(1, round(b_height / 3.0))

        confidence = float(pred.get("confidence", 1.0))
        fid = pred.get("detection_id") or f"rf_{uuid.uuid4().hex[:8]}"

        features.append({
            "type": "Feature",
            "id": f"b_{provider}_{idx}",
            "geometry": {
                "type": "Polygon",
                "coordinates": [ring],
            },
            "properties": {
                "type": "building",
                "name": f"Bina #{len(features) + 1}",
                "class": pred.get("class", "Building"),
                "height": b_height,
                "elevation": 0,
                "floors": floors,
                "ai_confidence": None if pred.get("prediction_source") == "color_heuristic" else round(confidence, 3),
                "heuristic_score": round(confidence, 3) if pred.get("prediction_source") == "color_heuristic" else None,
                "source": pred.get("prediction_source", provider),
                "height_source": "area_heuristic",
                "area_px": int(round(area_px)),
                "color": [0, 255, 157],
            },
        })

    # Dikdörtgenleştirme sonrası poligonların birbirinin üstüne binmesini kesin olarak engelle
    if processing_profile == "legacy" and len(features) > 1:
        from shapely.geometry import shape as _shape
        order_f = sorted(
            range(len(features)),
            key=lambda k: -float(features[k]["properties"].get("ai_confidence") or features[k]["properties"].get("heuristic_score") or 0),
        )
        kept_f = []
        kept_g = []
        kept_b = []
        for k in order_f:
            try:
                gk = _shape(features[k]["geometry"])
                if gk.is_empty or not gk.is_valid:
                    continue
                ak = max(gk.area, 1e-12)
                bx0, by0, bx1, by1 = gk.bounds
                if any(
                    not (bx1 < jx0 or bx0 > jx1 or by1 < jy0 or by0 > jy1)
                    and gk.intersects(gj)
                    and gk.intersection(gj).area / max(min(ak, gj.area), 1e-12) > 0.24
                    for gj, (jx0, jy0, jx1, jy1) in zip(kept_g, kept_b)
                ):
                    continue
                kept_f.append(features[k])
                kept_g.append(gk)
                kept_b.append((bx0, by0, bx1, by1))
            except Exception:
                continue
        for idx_f, feat in enumerate(kept_f):
            feat["properties"]["name"] = f"Bina #{idx_f + 1}"
        features = kept_f

    logger.info(f"Roboflow GeoJSON tamamlandı: {len(features)} bina poligonu üretildi.")

    return {
        "type": "FeatureCollection",
        "features": features,
        "meta": {
            "source": "roboflow",
            "model_id": ROBOFLOW_MODEL_ID,
            "building_count": len(features),
            "center": {"lat": center_lat, "lon": center_lon},
            "zoom": zoom,
            "bounds": bounds,
            "image_size": {"width": img_w, "height": img_h},
        },
    }


def regularize_pixel_polygon(raw_pts, simplify_epsilon=2.0):
    return preserve_polygon(raw_pts, simplify_epsilon)


def _legacy_regularize_pixel_polygon(raw_pts: list[list[float]], simplify_epsilon: float = 2.0) -> list[list[float]]:
    """
    AI modelinden dönen amip/yuvarlak hatlı bina maskelerini düz
    duvarlı, 90° dik açılı mimari poligonlara (Oriented Bounding Box / Orthogonal L-U Polygon)
    dönüştürür; tırtıklı ham konturlar sadeleştirilir.
    """
    if len(raw_pts) < 3:
        return raw_pts

    pts_np = np.array(raw_pts, dtype=np.float32).reshape((-1, 1, 2))
    area = float(cv2.contourArea(pts_np))
    if area <= 4.0:
        return raw_pts

    rect = cv2.minAreaRect(pts_np)
    (cx, cy), (rw, rh), angle = rect
    rect_area = float(rw * rh)
    if rect_area <= 1.0:
        return raw_pts

    rectangularity = area / rect_area

    # Yuvarlatılmış segmentasyon maskesinin dikdörtgen kutuyu şişirmesini önlemek için
    # alan korumalı sıkılaştırma faktörü uygula
    shrink = math.sqrt(max(0.82, min(1.0, rectangularity / 0.88)))
    tight_rect = ((cx, cy), (rw * shrink, rh * shrink), angle)

    # Normal dikdörtgen binalar veya küçük çatılar için doğrudan yönelimli dikdörtgen
    if rectangularity >= 0.62 or len(raw_pts) <= 6 or area < 1800.0:
        box = cv2.boxPoints(tight_rect)
        return [[float(p[0]), float(p[1])] for p in box]

    # L veya U tipi binalar için baskın çatı yöneliminde 90° dik açılı ızgara (Orthogonal Grid)
    from shapely.geometry import Polygon

    rad = math.radians(angle)
    cos_a, sin_a = math.cos(-rad), math.sin(-rad)
    pts_2d = pts_np.reshape(-1, 2) - np.array([cx, cy], dtype=np.float32)
    rot_pts = np.stack([
        pts_2d[:, 0] * cos_a - pts_2d[:, 1] * sin_a,
        pts_2d[:, 0] * sin_a + pts_2d[:, 1] * cos_a,
    ], axis=-1)

    grid_n = 10
    half_w, half_h = max(rw / 2.0, 1.0), max(rh / 2.0, 1.0)
    gx_pts = ((rot_pts[:, 0] + half_w) / max(rw, 1.0)) * grid_n
    gy_pts = ((rot_pts[:, 1] + half_h) / max(rh, 1.0)) * grid_n
    grid_mask = np.zeros((grid_n + 2, grid_n + 2), dtype=np.uint8)
    grid_poly = np.stack([gx_pts + 1.0, gy_pts + 1.0], axis=-1).round().astype(np.int32)
    cv2.fillPoly(grid_mask, [grid_poly], 255)

    cnts, _ = cv2.findContours(grid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cnt = max(cnts, key=cv2.contourArea)
        approx = cv2.approxPolyDP(cnt, 0.45, True).reshape(-1, 2)
        if 4 <= len(approx) <= 10:
            # Her kenarı tam yatay/dikey (90°) hizala
            ortho = [approx[0].astype(np.float32).copy()]
            for i in range(1, len(approx)):
                prev = ortho[-1]
                cur = approx[i].astype(np.float32).copy()
                if abs(cur[0] - prev[0]) < abs(cur[1] - prev[1]):
                    cur[0] = prev[0]
                else:
                    cur[1] = prev[1]
                if abs(cur[0] - prev[0]) > 0.1 or abs(cur[1] - prev[1]) > 0.1:
                    ortho.append(cur)
            if len(ortho) >= 4:
                # Son köşeyi ilk köşeyle 90° kapat
                first, last = ortho[0], ortho[-1]
                if abs(last[0] - first[0]) > 0.1 and abs(last[1] - first[1]) > 0.1:
                    ortho.append(np.array([first[0], last[1]], dtype=np.float32))
                ortho_np = np.array(ortho, dtype=np.float32)
                rot_ortho = np.stack([
                    ((ortho_np[:, 0] - 1.0) / grid_n) * rw - half_w,
                    ((ortho_np[:, 1] - 1.0) / grid_n) * rh - half_h,
                ], axis=-1)
                ortho_g = Polygon(rot_ortho).buffer(0)
                mask_g = Polygon(rot_pts).buffer(0)
                if (isinstance(ortho_g, Polygon) and ortho_g.is_valid and not ortho_g.is_empty
                        and len(ortho_g.exterior.coords) - 1 <= 10
                        and ortho_g.intersection(mask_g).area >= 0.58 * ortho_g.union(mask_g).area):
                    rot_coords = np.array(ortho_g.exterior.coords[:-1], dtype=np.float32)
                    cos_b, sin_b = math.cos(rad), math.sin(rad)
                    world_pts = np.stack([
                        rot_coords[:, 0] * cos_b - rot_coords[:, 1] * sin_b + cx,
                        rot_coords[:, 0] * sin_b + rot_coords[:, 1] * cos_b + cy,
                    ], axis=-1)
                    return [[float(p[0]), float(p[1])] for p in world_pts]

    box = cv2.boxPoints(tight_rect)
    return [[float(p[0]), float(p[1])] for p in box]


# ── Mapbox Vector Tiles (MVT) & OSM Hibrit Gerçek Bina Vektör Motoru ─────────
def _mvt_read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    res, shift = 0, 0
    while True:
        b = buf[pos]
        pos += 1
        res |= (b & 0x7F) << shift
        if not (b & 0x80):
            return res, pos
        shift += 7


def _mvt_parse_pbf(buf: bytes, start: int, end: int) -> dict[int, list]:
    pos = start
    fields: dict[int, list] = {}
    while pos < end:
        tag, pos = _mvt_read_varint(buf, pos)
        fnum, wtype = tag >> 3, tag & 7
        if wtype == 0:
            val, pos = _mvt_read_varint(buf, pos)
        elif wtype == 2:
            ln, pos = _mvt_read_varint(buf, pos)
            val = (pos, pos + ln)
            pos += ln
        elif wtype == 5:
            val = buf[pos:pos + 4]
            pos += 4
        elif wtype == 1:
            val = buf[pos:pos + 8]
            pos += 8
        else:
            break
        fields.setdefault(fnum, []).append(val)
    return fields


def _mvt_unpack_uints(buf: bytes, start: int, end: int) -> list[int]:
    out = []
    pos = start
    while pos < end:
        v, pos = _mvt_read_varint(buf, pos)
        out.append(v)
    return out


def _mvt_decode_rings(geom_ints: list[int], is_polygon: bool = True) -> list[list[tuple[int, int]]]:
    rings, cur = [], []
    x, y, i, n = 0, 0, 0, len(geom_ints)
    while i < n:
        cmd_len = geom_ints[i]
        i += 1
        cmd, count = cmd_len & 7, cmd_len >> 3
        if cmd == 1:  # MoveTo
            for _ in range(count):
                if cur:
                    rings.append(cur)
                    cur = []
                dx = (geom_ints[i] >> 1) ^ (-(geom_ints[i] & 1))
                dy = (geom_ints[i + 1] >> 1) ^ (-(geom_ints[i + 1] & 1))
                i += 2
                x += dx
                y += dy
                cur.append((x, y))
        elif cmd == 2:  # LineTo
            for _ in range(count):
                dx = (geom_ints[i] >> 1) ^ (-(geom_ints[i] & 1))
                dy = (geom_ints[i + 1] >> 1) ^ (-(geom_ints[i + 1] & 1))
                i += 2
                x += dx
                y += dy
                cur.append((x, y))
        elif cmd == 7:  # ClosePath
            if cur:
                if is_polygon and cur[0] != cur[-1]:
                    cur.append(cur[0])
                rings.append(cur)
                cur = []
    if cur:
        if is_polygon and cur[0] != cur[-1]:
            cur.append(cur[0])
        rings.append(cur)
    return rings


def _fetch_single_mvt_tile(args: tuple[int, int, int, str]) -> tuple[int, int, bytes | None]:
    """Tek bir MVT karosunu bellek önbelleğinden veya Mapbox CDN'den indirir."""
    import gzip
    z, xt, yt, mapbox_token = args
    cache_key = (z, xt, yt)
    cached = _MVT_TILE_CACHE.get(cache_key)
    if cached is not None:
        return xt, yt, cached

    url = f"https://api.mapbox.com/v4/mapbox.mapbox-streets-v8/{z}/{xt}/{yt}.mvt?access_token={mapbox_token}"
    try:
        r = _HTTP_SESSION.get(url, timeout=(3.0, 5.0))
        if r.status_code != 200:
            return xt, yt, None
        raw = gzip.decompress(r.content) if r.content[:2] == b"\x1f\x8b" else r.content
        if len(_MVT_TILE_CACHE) > 512:
            _MVT_TILE_CACHE.clear()
        _MVT_TILE_CACHE[cache_key] = raw
        return xt, yt, raw
    except Exception as e:
        logger.warning(f"MVT tile ({xt},{yt}) indirilemedi: {type(e).__name__}")
        return xt, yt, None


def fetch_mapbox_osm_vectors(bounds: dict, mapbox_token: str, include_status: bool = False):
    """
    Mapbox 'mapbox.mapbox-streets-v8' vektör karolarından (OpenStreetMap + Mapbox
    gerçek mimari bina ayak izleri ve yol ağı) Bounding Box içindeki tüm binaları
    ve yolları GeoJSON formatında paralel ve önbellekli olarak çeker.
    """
    import struct
    from shapely.geometry import Polygon, MultiPolygon, box as shapely_box
    from shapely.ops import unary_union
    from shapely.strtree import STRtree
    from shapely.validation import make_valid

    south = float(bounds["south"])
    west  = float(bounds["west"])
    north = float(bounds["north"])
    east  = float(bounds["east"])

    bbox_key = (south, west, north, east)
    now = time.monotonic()
    cached_entry = _VECTOR_BBOX_CACHE.get(bbox_key)
    if cached_entry is not None and (now - cached_entry[0]) < _VECTOR_CACHE_TTL_S:
        values = (copy.deepcopy(cached_entry[1]), copy.deepcopy(cached_entry[2]))
        return (*values, copy.deepcopy(cached_entry[3])) if include_status else values

    z = 16
    n = 2 ** z
    gx_west  = (west + 180.0) / 360.0 * n
    gx_east  = (east + 180.0) / 360.0 * n
    gy_north = (1.0 - math.asinh(math.tan(math.radians(north))) / math.pi) / 2.0 * n
    gy_south = (1.0 - math.asinh(math.tan(math.radians(south))) / math.pi) / 2.0 * n

    x0, x1 = int(gx_west), int(gx_east)
    y0, y1 = int(gy_north), int(gy_south)

    bbox_geom = shapely_box(west, south, east, north)
    by_id: dict[int, list[Polygon]] = {}
    heights_by_id: dict[int, float] = {}
    unkeyed_polys: list[Polygon] = []
    road_features: list[dict] = []
    seen_roads = set()

    tile_jobs = [
        (z, xt, yt, mapbox_token)
        for xt in range(min(x0, x1), max(x0, x1) + 1)
        for yt in range(min(y0, y1), max(y0, y1) + 1)
    ]

    # Tüm MVT karolarını paralel olarak indir
    fetched_tiles = list(_SHARED_POOL.map(_fetch_single_mvt_tile, tile_jobs))
    source_status = {"tiles": len(tile_jobs), "failed_tiles": sum(not raw for _, _, raw in fetched_tiles)}
    source_status["complete"] = source_status["failed_tiles"] == 0

    for xt, yt, raw in fetched_tiles:
        if not raw:
            continue

        top = _mvt_parse_pbf(raw, 0, len(raw))
        for ls, le in top.get(3, []):
            lf = _mvt_parse_pbf(raw, ls, le)
            if 1 not in lf:
                continue
            ns, ne = lf[1][0]
            lname = raw[ns:ne].decode("utf-8", errors="ignore")
            if lname not in ("building", "road"):
                continue

            extent = float(lf.get(5, [4096])[0])
            # Karo içi piksel sınırları (kenar taşmaları için 64 birim tolerans)
            tx_min = (gx_west - xt) * extent - 64.0
            tx_max = (gx_east - xt) * extent + 64.0
            ty_min = (gy_north - yt) * extent - 64.0
            ty_max = (gy_south - yt) * extent + 64.0

            keys = [raw[ks:ke].decode("utf-8", errors="ignore") for ks, ke in lf.get(3, [])]
            vals = []
            for vs, ve in lf.get(4, []):
                vf = _mvt_parse_pbf(raw, vs, ve)
                if 1 in vf:
                    s0, s1 = vf[1][0]
                    vals.append(raw[s0:s1].decode("utf-8", errors="ignore"))
                elif 2 in vf:
                    vals.append(struct.unpack("<f", vf[2][0])[0])
                elif 3 in vf:
                    vals.append(struct.unpack("<d", vf[3][0])[0])
                elif 4 in vf:
                    vals.append(vf[4][0])
                elif 5 in vf:
                    vals.append(vf[5][0])
                elif 7 in vf:
                    vals.append(bool(vf[7][0]))
                else:
                    vals.append(None)

            for fs, fe in lf.get(2, []):
                ff = _mvt_parse_pbf(raw, fs, fe)
                ftype = ff.get(3, [0])[0]
                if 4 not in ff:
                    continue
                gs, ge = ff[4][0]
                geom_ints = _mvt_unpack_uints(raw, gs, ge)

                # Tag'leri çöz
                props = {}
                if 2 in ff:
                    ts, te = ff[2][0]
                    t_ints = _mvt_unpack_uints(raw, ts, te)
                    for ti in range(0, len(t_ints) - 1, 2):
                        k_idx, v_idx = t_ints[ti], t_ints[ti + 1]
                        if k_idx < len(keys) and v_idx < len(vals):
                            props[keys[k_idx]] = vals[v_idx]

                if lname == "building" and ftype == 3:
                    if str(props.get("underground", "false")).lower() == "true":
                        continue
                    rings = _mvt_decode_rings(geom_ints, is_polygon=True)
                    if not rings:
                        continue
                    # Hızlı karo-içi Bounding Box ön elemesi (trigonometri ve Shapely öncesi)
                    first_ring = rings[0]
                    xs_r = [pt[0] for pt in first_ring]
                    ys_r = [pt[1] for pt in first_ring]
                    if max(xs_r) < tx_min or min(xs_r) > tx_max or max(ys_r) < ty_min or min(ys_r) > ty_max:
                        continue

                    fid = ff.get(1, [None])[0]
                    b_height = float(props.get("height") or 0.0)

                    current_outer = None
                    current_holes = []
                    polys_in_feat = []

                    for ring in rings:
                        if len(ring) < 4:
                            continue
                        signed_area = sum(
                            ring[k][0] * ring[k + 1][1] - ring[k + 1][0] * ring[k][1]
                            for k in range(len(ring) - 1)
                        )
                        geo_ring = []
                        for tx, ty in ring:
                            gx = xt + tx / extent
                            gy = yt + ty / extent
                            plon = gx / n * 360.0 - 180.0
                            plat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * gy / n))))
                            geo_ring.append((plon, plat))

                        if signed_area > 0:  # Exterior ring
                            if current_outer is not None:
                                try:
                                    p = Polygon(current_outer, current_holes)
                                    if not p.is_valid:
                                        p = make_valid(p)
                                    polys_in_feat.append(p)
                                except Exception:
                                    pass
                            current_outer = geo_ring
                            current_holes = []
                        else:  # Interior hole
                            if current_outer is not None:
                                current_holes.append(geo_ring)

                    if current_outer is not None:
                        try:
                            p = Polygon(current_outer, current_holes)
                            if not p.is_valid:
                                p = make_valid(p)
                            polys_in_feat.append(p)
                        except Exception:
                            pass

                    for p in polys_in_feat:
                        if p.is_empty:
                            continue
                        if fid is not None:
                            by_id.setdefault(fid, []).append(p)
                            if b_height > 0:
                                heights_by_id[fid] = max(heights_by_id.get(fid, 0.0), b_height)
                        else:
                            unkeyed_polys.append(p)

                elif lname == "road" and ftype == 2:
                    lines = _mvt_decode_rings(geom_ints, is_polygon=False)
                    for line in lines:
                        if len(line) < 2:
                            continue
                        geo_line = []
                        for tx, ty in line:
                            gx = xt + tx / extent
                            gy = yt + ty / extent
                            plon = round(gx / n * 360.0 - 180.0, 7)
                            plat = round(math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * gy / n)))), 7)
                            if west <= plon <= east and south <= plat <= north:
                                if not geo_line or geo_line[-1] != [plon, plat]:
                                    geo_line.append([plon, plat])
                        if len(geo_line) >= 2:
                            sig = (tuple(geo_line[0]), tuple(geo_line[-1]), len(geo_line))
                            if sig not in seen_roads:
                                seen_roads.add(sig)
                                road_class = str(props.get("class", "street"))
                                road_name = props.get("name") or props.get("name_tr") or props.get("ref") or None
                                width = HIGHWAY_WIDTH_MAP.get(road_class, 3)
                                road_features.append({
                                    "type": "Feature",
                                    "id": f"r_{len(road_features) + 1}",
                                    "geometry": {"type": "LineString", "coordinates": geo_line},
                                    "properties": {
                                        "type": "road",
                                        "highway": road_class,
                                        "class": road_class,
                                        "name": road_name,
                                        "oneway": str(props.get("oneway", "false")),
                                        "width": width,
                                        "color": [255, 140, 0],
                                    },
                                })

    # Karo sınırlarında bölünen aynı ID'li binaları tek parça olarak birleştir (unary_union)
    merged_items: list[tuple[Polygon, float]] = []
    for fid, plist in by_id.items():
        try:
            u = unary_union(plist) if len(plist) > 1 else plist[0]
            if not u.is_valid:
                u = make_valid(u)
            h_val = heights_by_id.get(fid, 0.0)
            if isinstance(u, Polygon):
                merged_items.append((u, h_val))
            elif isinstance(u, MultiPolygon):
                for sub in u.geoms:
                    if isinstance(sub, Polygon):
                        merged_items.append((sub, h_val))
        except Exception:
            continue

    for p in unkeyed_polys:
        if isinstance(p, Polygon):
            merged_items.append((p, 0.0))

    # Yalnızca 1280x1280 Bounding Box içinde kalan binaları filtrele
    lat_mid = (north + south) / 2.0
    m2_scale = (111_320.0 ** 2) * math.cos(math.radians(lat_mid))

    valid_polys: list[tuple[Polygon, float, float]] = []
    for poly, h_val in merged_items:
        minx, miny, maxx, maxy = poly.bounds
        if maxx < west or minx > east or maxy < south or miny > north:
            continue
        if minx < west or maxx > east or miny < south or maxy > north:
            c = poly.centroid
            if not (west <= c.x <= east and south <= c.y <= north):
                if not poly.intersects(bbox_geom):
                    continue
                clipped = poly.intersection(bbox_geom)
                if not isinstance(clipped, Polygon) or clipped.is_empty:
                    continue
                poly = clipped

        area_m2 = poly.area * m2_scale
        if area_m2 < 14.0:
            continue
        valid_polys.append((poly, h_val, area_m2))

    # Büyükten küçüğe sırala ve STRtree uzamsal indeksiyle alt parçaları (building:part) O(N log N) sürede ele
    valid_polys.sort(key=lambda x: -x[2])
    if len(valid_polys) > 1:
        all_geoms = [vp[0] for vp in valid_polys]
        tree = STRtree(all_geoms)
        suppressed: set[int] = set()
        for i, (poly_i, _, _) in enumerate(valid_polys):
            if i in suppressed:
                continue
            for j in tree.query(poly_i):
                j_idx = int(j)
                if j_idx <= i or j_idx in suppressed:
                    continue
                poly_j = valid_polys[j_idx][0]
                if poly_i.intersects(poly_j):
                    inter = poly_i.intersection(poly_j).area
                    if inter / max(poly_j.area, 1e-12) > 0.75:
                        suppressed.add(j_idx)
        kept_polys = [vp for idx, vp in enumerate(valid_polys) if idx not in suppressed]
    else:
        kept_polys = list(valid_polys)

    building_features: list[dict] = []
    for idx, (poly, h_val, area_m2) in enumerate(kept_polys):
        ext_coords = [[round(float(x), 7), round(float(y), 7)] for x, y in poly.exterior.coords]
        if not ext_coords or len(ext_coords) < 4:
            continue
        if ext_coords[0] != ext_coords[-1]:
            ext_coords.append([ext_coords[0][0], ext_coords[0][1]])

        height = round(h_val, 1) if h_val >= 4.0 else round(min(45.0, max(9.0, 9.0 + math.sqrt(area_m2) * 0.45)), 1)
        floors = max(1, round(height / 3.0))

        building_features.append({
            "type": "Feature",
            "id": f"b_osm_{idx + 1}",
            "geometry": {
                "type": "Polygon",
                "coordinates": [ext_coords],
            },
            "properties": {
                "type": "building",
                "name": f"Bina #{idx + 1}",
                "class": "Building",
                "height": height,
                "elevation": 0,
                "floors": floors,
                "source": "osm",
                "area_m2": int(round(area_m2)),
                "color": [0, 255, 157],
            },
        })

    if len(_VECTOR_BBOX_CACHE) > 64:
        _VECTOR_BBOX_CACHE.clear()
    if source_status["complete"]:
        _VECTOR_BBOX_CACHE[bbox_key] = (now, copy.deepcopy(building_features), copy.deepcopy(road_features), source_status)

    values = (building_features, road_features)
    return (*values, source_status) if include_status else values


def estimate_osm_offset(
    osm_buildings: list[dict],
    ai_buildings: list[dict],
    bounds: dict,
    grid: int = 640,
    max_shift_m: float = 12.0,
    min_gain: float = 1.03,
) -> tuple[float, float, float, float] | None:
    """
    OSM bina ayak izleri ile uydu görüntüsündeki çatılar arasındaki global kaymayı
    tahmin eder. Uydu görüntüsü tam dikey çekilmediğinde çatılar zemine göre
    metrelerce kayar (Bartın'da ~5 m); OSM ise zemindeki ayak izini çizer.
    AI poligonları görüntüden geldiği için çatı konumunu verir; iki maskenin
    çapraz korelasyonu (FFT) bu kaymayı ölçer.

    Döndürür: (dlon, dlat, east_m, north_m) veya güvenilir bir kayma yoksa None.
    """
    if len(ai_buildings) < 10 or not osm_buildings:
        return None

    north, south = float(bounds["north"]), float(bounds["south"])
    east, west = float(bounds["east"]), float(bounds["west"])
    sx = grid / (east - west)
    sy = grid / (north - south)

    def _mask(features):
        m = np.zeros((grid, grid), np.float32)
        for f in features:
            ring = f["geometry"]["coordinates"][0]
            pts = np.array([[(lon - west) * sx, (north - lat) * sy] for lon, lat in ring], np.int32)
            cv2.fillPoly(m, [pts], 1.0)
        return m

    ai_mask, osm_mask = _mask(ai_buildings), _mask(osm_buildings)
    min_area = min(ai_mask.sum(), osm_mask.sum())
    if min_area <= 0:
        return None

    # Sıfır dolgulu FFT: cc[d] = OSM maskesi d kadar kaydırıldığında örtüşen piksel sayısı
    n = 2 * grid
    cc = np.real(np.fft.ifft2(np.fft.fft2(ai_mask, (n, n)) * np.conj(np.fft.fft2(osm_mask, (n, n)))))
    cc = np.fft.fftshift(cc)

    lat_mid = (north + south) / 2.0
    m_per_px = (east - west) * 111_320.0 * math.cos(math.radians(lat_mid)) / grid
    r = max(1, int(max_shift_m / m_per_px))
    win = cc[grid - r:grid + r + 1, grid - r:grid + r + 1]
    iy, ix = np.unravel_index(int(np.argmax(win)), win.shape)
    dy, dx = int(iy) - r, int(ix) - r

    best, base = float(win[iy, ix]), float(cc[grid, grid])
    # Pencere kenarındaki tepe (aramanın dışına taşan) veya zayıf eşleşme güvenilir değil
    if (dx, dy) == (0, 0) or abs(dx) == r or abs(dy) == r:
        return None
    if best < max(base, 1.0) * min_gain or best < 0.15 * min_area:
        return None

    return dx / sx, -dy / sy, round(dx * m_per_px, 1), round(-dy * m_per_px, 1)


def shift_features(features: list[dict], dlon: float, dlat: float) -> list[dict]:
    """Poligon özelliklerini yerinde (dlon, dlat) kadar kaydırır."""
    for f in features:
        f["geometry"]["coordinates"] = [
            [[round(x + dlon, 7), round(y + dlat, 7)] for x, y in ring]
            for ring in f["geometry"]["coordinates"]
        ]
    return features


# OSM'in boş bıraktığı yerlere eklenecek AI binaları için eşikler
AI_GAPFILL_MIN_CONF      = 0.35  # Eklenecek AI binası için asgari güven
AI_GAPFILL_MAX_OSM_COVER = 0.35  # AI poligonunun en fazla bu kadarı OSM ile örtülüyse OSM'de eksik sayılır
AI_MATCH_MIN_OVERLAP     = 0.30  # OSM binasına AI güveni aktarmak için asgari örtüşme (küçük poligona göre)
OSM_FRAGMENT_INSIDE      = 0.70  # AI binasının içinde kalan OSM poligonu, aynı binanın kısmi çizimidir


def merge_hybrid_buildings(
    osm_buildings: list[dict],
    roboflow_buildings: list[dict],
    min_gapfill_conf: float = AI_GAPFILL_MIN_CONF,
    osm_roads: list[dict] | None = None,
    bounds: dict | None = None,
    processing_profile: str = "conservative",
) -> list[dict]:
    """
    Hibrit Bina Birleştirici (OSM/Mapbox Gerçek Mimari Vektörler + Roboflow AI):
    - OSM/Mapbox'tan gelen bina poligonlarını temel alır.
    - Roboflow AI poligonlarıyla anlamlı ölçüde örtüşen OSM binalarına gerçek AI
      güven skorunu (ai_confidence) aktarır.
    - AI poligonunun büyük kısmı OSM'de boşsa (OSM kapsamının zayıf olduğu
      İstanbul dışı şehirler) binayı ekler; yol koridorlarıyla veya mevcut binalarla
      üst üste binmesini engeller.
    """
    if not osm_buildings:
        return [
            f for f in roboflow_buildings
            if float(f.get("properties", {}).get("ai_confidence") or 0) >= min_gapfill_conf
        ]

    from shapely.geometry import Polygon, MultiPolygon, shape
    from shapely.strtree import STRtree

    osm_geoms = []
    valid_osm = []
    for f in osm_buildings:
        try:
            g = shape(f["geometry"])
            if g.is_valid and not g.is_empty:
                osm_geoms.append(g)
                valid_osm.append(f)
        except Exception:
            continue

    rf_geoms = []
    rf_confs = []
    rf_feats = []
    for rf in roboflow_buildings:
        try:
            g = shape(rf["geometry"])
            if g.is_valid and not g.is_empty:
                rf_geoms.append(g)
                rf_confs.append(float(rf.get("properties", {}).get("ai_confidence") or 0))
                rf_feats.append(rf)
        except Exception:
            continue

    if not osm_geoms or not rf_geoms:
        return valid_osm

    road_corridor = _build_road_corridor_union(osm_roads, bounds) if (osm_roads and bounds) else None
    osm_tree = STRtree(osm_geoms)
    best_conf = [0.0] * len(osm_geoms)
    drop_osm: set[int] = set()

    # Zoom-16 MVT karolarından gelen genelleştirilmiş çok-binalı kaba blokları tespit et
    # (Özgün Overpass OSM binalarına asla dokunma)
    rf_tree = STRtree(rf_geoms)
    for i, (osm_f, osm_g) in enumerate(zip(valid_osm, osm_geoms)):
        props = osm_f.get("properties") or {}
        if props.get("geometry_source") == "overpass" or str(osm_f.get("id", "")).startswith("osm_"):
            continue
        try:
            n_verts = len(osm_g.exterior.coords) - 1 if isinstance(osm_g, Polygon) else 20
            mrr_a = max(osm_g.minimum_rotated_rectangle.area, 1e-12)
            rect_ratio = osm_g.area / mrr_a
            if processing_profile == "legacy" and n_verts >= 14 and rect_ratio < 0.56:
                inside_rf = 0
                for rj in rf_tree.query(osm_g):
                    rg = rf_geoms[int(rj)]
                    if rg.area < 0.55 * osm_g.area and osm_g.intersects(rg):
                        if osm_g.intersection(rg).area >= 0.55 * rg.area:
                            inside_rf += 1
                if inside_rf >= 2:
                    drop_osm.add(i)
        except Exception:
            pass

    candidates: list[tuple[float, dict, object]] = []

    for rf_f, rf_g, rf_c in zip(rf_feats, rf_geoms, rf_confs):
        rf_area = max(rf_g.area, 1e-12)
        overlaps = []
        for hi in osm_tree.query(rf_g):
            i = int(hi)
            if i in drop_osm:
                continue
            if rf_g.intersects(osm_geoms[i]):
                inter = rf_g.intersection(osm_geoms[i]).area
                if inter > 0:
                    overlaps.append((i, inter))

        # 1. Anlamlı örtüşen OSM binasına AI güven skorunu aktar (sadece kenar teması yetmez)
        for i, inter in overlaps:
            if inter / max(min(osm_geoms[i].area, rf_area), 1e-12) >= AI_MATCH_MIN_OVERLAP:
                best_conf[i] = max(best_conf[i], rf_c)

        # 2. AI binasının çoğu OSM'de boşsa ekle; ancak mevcut bir OSM binasıyla üst üste binmesine izin verme
        if rf_c < min_gapfill_conf:
            continue

        osm_cover = sum(inter for _, inter in overlaps) / rf_area
        if osm_cover >= 0.25:
            continue

        has_blocking_osm = False
        for i, inter in overlaps:
            small_ratio = inter / max(min(osm_geoms[i].area, rf_area), 1e-12)
            if small_ratio >= 0.24:
                if (valid_osm[i].get("properties", {}).get("geometry_source") != "overpass"
                        and not str(valid_osm[i].get("id", "")).startswith("osm_")
                        and inter / max(osm_geoms[i].area, 1e-12) >= OSM_FRAGMENT_INSIDE
                        and rf_area <= 1.6 * osm_geoms[i].area):
                    drop_osm.add(i)
                else:
                    has_blocking_osm = True
                    break
        if has_blocking_osm:
            continue

        # Yol koridoru kontrolü
        cand_g = rf_g
        if road_corridor is not None and cand_g.intersects(road_corridor):
            r_inter = cand_g.intersection(road_corridor).area
            if r_inter > 0.42 * cand_g.area:
                continue
            if r_inter > 0.08 * cand_g.area:
                diff_g = cand_g.difference(road_corridor).buffer(0)
                if isinstance(diff_g, MultiPolygon):
                    diff_g = max(diff_g.geoms, key=lambda q: q.area)
                if isinstance(diff_g, Polygon) and not diff_g.is_empty and diff_g.area >= 0.52 * cand_g.area:
                    mrr = diff_g.minimum_rotated_rectangle
                    if (
                        isinstance(mrr, Polygon)
                        and not mrr.is_empty
                        and (not road_corridor.intersects(mrr) or mrr.intersection(road_corridor).area <= 0.22 * mrr.area)
                    ):
                        cand_g = mrr
                    elif len(diff_g.exterior.coords) >= 5:
                        cand_g = diff_g

        candidates.append((rf_c, rf_f, cand_g))

    for i, osm_f in enumerate(valid_osm):
        osm_f["properties"]["ai_verified"] = best_conf[i] > 0
        osm_f["properties"]["ai_confidence"] = round(best_conf[i], 3) if best_conf[i] > 0 else None

    merged = [f for i, f in enumerate(valid_osm) if i not in drop_osm]
    accepted_geoms = [osm_geoms[i] for i in range(len(valid_osm)) if i not in drop_osm]
    accepted_bounds = [g.bounds for g in accepted_geoms]

    # Yüksek güvenliden düşüğe sırala ve eklenen AI binalarının kendi aralarında veya OSM ile çakışmasını önle
    candidates.sort(key=lambda x: -x[0])
    for rf_c, rf_f, cand_g in candidates:
        c_area = max(cand_g.area, 1e-12)
        cx0, cy0, cx1, cy1 = cand_g.bounds
        if any(
            not (cx1 < ax0 or cx0 > ax1 or cy1 < ay0 or cy0 > ay1)
            and cand_g.intersects(ag)
            and cand_g.intersection(ag).area / max(min(c_area, ag.area), 1e-12) > 0.22
            for ag, (ax0, ay0, ax1, ay1) in zip(accepted_geoms, accepted_bounds)
        ):
            continue
        ext_ring = [[round(float(x), 7), round(float(y), 7)] for x, y in cand_g.exterior.coords]
        if ext_ring and ext_ring[0] != ext_ring[-1]:
            ext_ring.append([ext_ring[0][0], ext_ring[0][1]])
        if len(ext_ring) < 4:
            continue

        rf_copy = dict(rf_f)
        rf_copy["id"] = f"b_ai_{len(merged) + 1}"
        rf_copy["geometry"] = {"type": "Polygon", "coordinates": [ext_ring]}
        rf_copy["properties"] = dict(rf_f.get("properties", {}))
        rf_copy["properties"]["name"] = f"Bina #{len(merged) + 1}"
        rf_copy["properties"]["ai_verified"] = True
        merged.append(rf_copy)
        accepted_geoms.append(cand_g)
        accepted_bounds.append((cx0, cy0, cx1, cy1))

    return merged


# ── Overpass API / Mapbox OSM Vektör Yol Ağı Çekici ──────────────────────────
OVERPASS_ROADS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

DEFAULT_MAPBOX_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN", "")

HIGHWAY_WIDTH_MAP = {
    "motorway":       6,
    "trunk":          5,
    "primary":        5,
    "secondary":      4,
    "tertiary":       4,
    "residential":    3,
    "living_street":  3,
    "unclassified":   3,
    "street":         3,
    "pedestrian":     2.5,
    "service":        2,
    "footway":        1.5,
    "path":           1.5,
}


def fetch_osm_roads_overpass(
    bounds: dict,
    mapbox_token: str | None = DEFAULT_MAPBOX_TOKEN,
    timeout: float = 2.5,
) -> dict:
    """
    Verilen Bounding Box (south, west, north, east) koordinatları içindeki
    tüm yol ağlarını (highways, streets) çeker ve deck.gl PathLayer /
    GeoJsonLayer uyumlu standart GeoJSON (LineString) FeatureCollection formatına dönüştürür.

    Hız Optimizasyonu:
    - Mapbox erişim anahtarı mevcutsa öncelikle Mapbox CDN üzerindeki OpenStreetMap
      vektör karolarından (MVT) ~0.2 saniyede (önbellekten <1ms) yolları çeker.
    - Mapbox MVT boş dönerse veya token yoksa Overpass API sunucularına başvurur.
    """
    south = float(bounds["south"])
    west  = float(bounds["west"])
    north = float(bounds["north"])
    east  = float(bounds["east"])

    road_features: list[dict] = []
    source_used = "mapbox_osm_mvt"
    last_error = None

    # 1. Hızlı Yol: Mapbox CDN üzerindeki OSM Vektör Karolarından (MVT) anında çek
    if mapbox_token and not mapbox_token.startswith("TODO"):
        try:
            _, mvt_roads = fetch_mapbox_osm_vectors(bounds, mapbox_token)
            if mvt_roads:
                for r in mvt_roads:
                    props = r.setdefault("properties", {})
                    props["color"] = [255, 140, 0]
                    props.setdefault("highway", props.get("class", "street"))
                    props.setdefault("width", HIGHWAY_WIDTH_MAP.get(props["highway"], 3))
                road_features = mvt_roads
        except Exception as mvt_err:
            last_error = f"MVT: {mvt_err}"
            logger.warning(f"Mapbox MVT yol çekimi başarısız, Overpass deneniyor: {mvt_err}")

    # 2. Yedek Yol: MVT boş döndüyse Overpass API üzerinden çek
    if not road_features:
        source_used = "overpass_osm"
        overpass_query = (
            f"[out:json][timeout:{max(1, int(timeout))}];"
            f"("
            f'way["highway"]({south},{west},{north},{east});'
            f");"
            f"out geom;"
        )

        elements = None
        for url in OVERPASS_ROADS_URLS:
            try:
                resp = _get_http_session().post(
                    url,
                    data={"data": overpass_query},
                    timeout=(1.5, timeout),
                    headers={"User-Agent": "GeoSim-DigitalTwin/3.0"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    cand_elements = data.get("elements", [])
                    if cand_elements:
                        elements = cand_elements
                        break
                    last_error = data.get("remark") or f"{url} -> 0 elements"
                else:
                    last_error = f"{url} -> HTTP {resp.status_code}"
            except Exception as exc:
                last_error = f"{url} -> {exc}"
                continue

        if elements is not None:
            for idx, el in enumerate(elements):
                if el.get("type") != "way":
                    continue
                geom = el.get("geometry")
                if not geom or len(geom) < 2:
                    continue

                coords: list[list[float]] = []
                for pt in geom:
                    lon = round(float(pt["lon"]), 7)
                    lat = round(float(pt["lat"]), 7)
                    if not coords or coords[-1] != [lon, lat]:
                        coords.append([lon, lat])

                if len(coords) < 2:
                    continue

                tags = el.get("tags", {}) or {}
                highway_type = str(tags.get("highway", "residential"))
                road_name = tags.get("name") or tags.get("ref") or None
                width = HIGHWAY_WIDTH_MAP.get(highway_type, 3)

                road_features.append({
                    "type": "Feature",
                    "id": f"osm_road_{el.get('id', idx + 1)}",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coords,
                    },
                    "properties": {
                        "type": "road",
                        "osm_id": el.get("id"),
                        "highway": highway_type,
                        "class": highway_type,
                        "name": road_name,
                        "oneway": tags.get("oneway", "no"),
                        "lanes": tags.get("lanes"),
                        "surface": tags.get("surface"),
                        "width": width,
                        "color": [255, 140, 0],
                    },
                })

    return {
        "type": "FeatureCollection",
        "features": road_features,
        "meta": {
            "source": source_used,
            "total_roads": len(road_features),
            "bounds": {
                "south": south,
                "west": west,
                "north": north,
                "east": east,
            },
            "overpass_error": last_error if not road_features else None,
        },
    }
