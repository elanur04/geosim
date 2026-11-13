"""Yerel, sürümlü çalışma alanları. Kaydedilen düzeltmeler model çıktısından bağımsızdır."""
import copy
import json
import math
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from shapely.geometry import box, shape
from shapely.errors import ShapelyError

ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("GEOSIM_REGIONS_DB", str(ROOT / "data" / "regions.sqlite3")))
router = APIRouter(prefix="/api/regions", tags=["Saved study areas"])


class RegionSave(BaseModel):
    revision: int = Field(ge=0)
    name: str = Field(min_length=1, max_length=160)
    collection: dict
    coverage_reviewed: bool = False


@contextmanager
def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=10)
    db.execute("CREATE TABLE IF NOT EXISTS regions (id TEXT, revision INTEGER, payload TEXT NOT NULL, PRIMARY KEY(id, revision))")
    try:
        with db:
            yield db
    finally:
        db.close()


def _check_id(region_id):
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", region_id):
        raise HTTPException(422, "Geçersiz bölge kimliği.")


def validate_collection(collection):
    if collection.get("type") != "FeatureCollection" or not isinstance(collection.get("features"), list):
        raise HTTPException(422, "GeoJSON FeatureCollection gerekli.")
    if len(collection["features"]) > 30000:
        raise HTTPException(422, "Bir çalışma alanı en fazla 30000 nesne içerebilir.")
    result = copy.deepcopy(collection)
    meta = result.get("meta")
    if not isinstance(meta, dict):
        raise HTTPException(422, "Çalışma alanı meta verisi gerekli.")
    bounds = meta.get("bounds", {})
    try:
        s, w, n, e = [float(bounds[k]) for k in ("south", "west", "north", "east")]
        if not all(math.isfinite(v) for v in (s, w, n, e)) or not (-85 <= s < n <= 85 and -180 <= w < e <= 180):
            raise ValueError()
    except (KeyError, TypeError, ValueError):
        raise HTTPException(422, "Geçerli çalışma alanı sınırları gerekli.")
    area = box(w, s, e, n)
    ids = set()
    for feature in result["features"]:
        if not isinstance(feature, dict):
            raise HTTPException(422, "Geçersiz nesne.")
        if feature.get("type") != "Feature":
            raise HTTPException(422, "Her nesne GeoJSON Feature olmalı.")
        fid = feature.get("id")
        if not isinstance(fid, str) or not fid or fid in ids:
            raise HTTPException(422, "Nesne kimlikleri boş olamaz ve tekrarlanamaz.")
        ids.add(fid)
        props = feature.get("properties")
        if not isinstance(props, dict) or props.get("type") not in ("building", "road"):
            raise HTTPException(422, "Yalnızca bina ve yol nesneleri kaydedilebilir.")
        try:
            raw = feature["geometry"]
            if props["type"] == "building":
                polygons = [raw["coordinates"]] if raw["type"] == "Polygon" else raw["coordinates"]
                if any(len(ring) < 4 or ring[0] != ring[-1] for polygon in polygons for ring in polygon):
                    raise ValueError()
            geometry = shape(feature["geometry"])
            expected = ("Polygon", "MultiPolygon") if props["type"] == "building" else ("LineString", "MultiLineString")
            if geometry.geom_type not in expected or geometry.is_empty or not geometry.is_valid:
                raise ValueError()
            gx0, gy0, gx1, gy1 = geometry.bounds
            if not all(math.isfinite(v) for v in geometry.bounds) or not (-180 <= gx0 <= gx1 <= 180 and -85 <= gy0 <= gy1 <= 85):
                raise ValueError()
            if not geometry.intersects(area):
                raise ValueError()
        except (KeyError, IndexError, TypeError, ValueError, ShapelyError):
            raise HTTPException(422, f"{fid}: geçersiz veya çalışma alanı dışında geometri.")
        # Arayüze ait yardımcı özellikler kaydedilmez.
        feature["properties"] = {k: v for k, v in props.items() if not k.startswith("_")}
        if props.get("review_status") not in ("reviewed", "unreviewed"):
            feature["properties"]["review_status"] = "unreviewed"
        dimension = "height" if props["type"] == "building" else "width"
        value = props.get(dimension)
        if value is not None:
            try:
                if isinstance(value, bool) or not 0 < float(value) <= 1000:
                    raise ValueError()
                feature["properties"][dimension] = float(value)
            except (ValueError, TypeError):
                raise HTTPException(422, f"{fid}: {dimension} pozitif, sonlu bir sayı olmalı.")
    return result


def quality_report(collection, coverage_reviewed=False):
    features = collection["features"]
    buildings = sum(f["properties"]["type"] == "building" for f in features)
    roads = len(features) - buildings
    unreviewed = sum(f["properties"].get("review_status") != "reviewed" for f in features)
    unknown_heights = sum(f["properties"]["type"] == "building" and
                          (f["properties"].get("height_source") not in ("osm", "manual") or
                           not f["properties"].get("height")) for f in features)
    issues = []
    if not buildings:
        issues.append("Çalışma alanında bina yok.")
    if not roads:
        issues.append("Çalışma alanında yol yok.")
    if unreviewed:
        issues.append(f"{unreviewed} nesnenin konumu ve sınırları henüz denetlenmedi.")
    if not coverage_reviewed:
        issues.append("Çalışma alanının tamamı eksik bina ve yollar için denetlenmedi.")
    return {"buildings": buildings, "roads": roads, "unreviewed": unreviewed,
            "unknown_heights": unknown_heights, "coverage_reviewed": coverage_reviewed,
            "geometry_review_complete": not issues, "issues": issues,
            "simulation_ready": False,
            "simulation_note": "Simülasyon türü ve gerektirdiği yol ağı / yapı verileri ayrıca doğrulanmalıdır."}


def _load(db, region_id):
    row = db.execute("SELECT payload FROM regions WHERE id=? ORDER BY revision DESC LIMIT 1", (region_id,)).fetchone()
    return json.loads(row[0]) if row else None


def _seed_bartin(db):
    if _load(db, "bartin") is not None:
        return
    path = ROOT / "verification" / "bartin-source.geojson"
    if not path.exists():
        return
    collection = validate_collection(json.loads(path.read_text(encoding="utf-8")))
    payload = {"id": "bartin", "name": "Bartın Merkez", "revision": 1,
               "updated_at": datetime.now(timezone.utc).isoformat(), "collection": collection,
               "quality": quality_report(collection)}
    db.execute("INSERT OR IGNORE INTO regions VALUES(?,?,?)", ("bartin", 1, json.dumps(payload, ensure_ascii=False)))


@router.get("")
def list_regions():
    with _connect() as db:
        _seed_bartin(db)
        rows = db.execute("SELECT payload FROM regions r WHERE revision=(SELECT MAX(revision) FROM regions WHERE id=r.id)").fetchall()
    return {"regions": [{k: json.loads(row[0])[k] for k in ("id", "name", "revision", "updated_at", "quality")} for row in rows]}


@router.get("/{region_id}")
def get_region(region_id: str, revision: int | None = Query(default=None, ge=1)):
    _check_id(region_id)
    with _connect() as db:
        if region_id == "bartin":
            _seed_bartin(db)
        if revision is None:
            payload = _load(db, region_id)
        else:
            row = db.execute("SELECT payload FROM regions WHERE id=? AND revision=?", (region_id, revision)).fetchone()
            payload = json.loads(row[0]) if row else None
    if payload is None:
        raise HTTPException(404, "Kayıtlı bölge bulunamadı.")
    return payload


@router.put("/{region_id}")
def save_region(region_id: str, req: RegionSave):
    _check_id(region_id)
    if not req.name.strip():
        raise HTTPException(422, "Çalışma alanı adı boş olamaz.")
    collection = validate_collection(req.collection)
    # Özellik ve meta verilerde de NaN/Infinity reddedilir.
    try:
        json.dumps(collection, allow_nan=False)
    except ValueError:
        raise HTTPException(422, "Sonlu olmayan sayılar kaydedilemez.")
    with _connect() as db:
        db.execute("BEGIN IMMEDIATE")
        current = _load(db, region_id)
        revision = current["revision"] if current else 0
        if req.revision != revision:
            raise HTTPException(409, "Bölge başka bir oturumda değişti. Güncel kaydı yeniden açın.")
        payload = {"id": region_id, "name": req.name.strip(), "revision": revision + 1,
                   "updated_at": datetime.now(timezone.utc).isoformat(), "collection": collection,
                   "quality": quality_report(collection, req.coverage_reviewed)}
        db.execute("INSERT INTO regions VALUES(?,?,?)", (region_id, revision + 1, json.dumps(payload, ensure_ascii=False)))
    return payload
