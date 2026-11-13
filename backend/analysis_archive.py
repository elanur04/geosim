"""Analiz girdilerini (görüntü, OSM, model çıktısı) sağlama toplamıyla arşivler; çevrimdışı karşılaştırmada kullanılır."""
import hashlib
import io
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parent / "data" / "analysis_runs"
router = APIRouter(prefix="/api/analysis-runs", tags=["Reproducible analysis"])
ARTIFACTS = {"image.png": "image/png", "osm.geojson": "application/geo+json",
             "ai.geojson": "application/geo+json", "hybrid.geojson": "application/geo+json",
             "model-output.json": "application/json", "ai-raw.geojson": "application/geo+json"}
PROCESSING_CODE_SHA256 = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                          for name in ("vectorize.py", "osm_source.py", "api.py", "prediction_pipeline.py", "building_models.py")}


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")


def archive_analysis(*, request, payload, image, osm_features, ai_features, model_output):
    now = datetime.now(timezone.utc)
    run_id = f"{now.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:12]}"
    directory = ROOT / run_id
    directory.mkdir(parents=True, exist_ok=False)
    artifacts = {}

    def put(name, raw):
        (directory / name).write_bytes(raw)
        artifacts[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}

    meta = payload["meta"]
    common = {"bounds": meta["bounds"], "analysis_run_id": run_id, "role": "prediction",
              "captured_at": now.isoformat(), "imagery_acquisition_date": None}
    for name, features in (("osm", osm_features), ("ai", ai_features), ("hybrid", payload["features"])):
        put(f"{name}.geojson", _json_bytes({"type": "FeatureCollection", "features": features,
                                           "meta": {**common, "method": name}}))
    if "raw_predictions" in model_output:
        from vectorize import roboflow_predictions_to_geojson
        raw_output = {**model_output, "predictions": model_output["raw_predictions"]}
        raw_geojson = roboflow_predictions_to_geojson(raw_output, bounds=meta["bounds"], processing_profile="raw")
        raw_geojson["meta"] = {**common, "method": "ai_raw", "duplicates_removed": False}
        put("ai-raw.geojson", _json_bytes(raw_geojson))
    put("model-output.json", _json_bytes(model_output))
    if image is not None:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        put("image.png", buffer.getvalue())
    errors = {k: meta.get(k) for k in ("roboflow_error", "failed_tiles") if meta.get(k)}
    if meta.get("osm_input_complete") is not True:
        errors["osm"] = "source_incomplete"
    if image is None:
        errors["image"] = "unavailable"
    if not meta.get("tiles"):
        errors["inference"] = "not_completed"
    manifest = {"schema_version": 2, "id": run_id, "captured_at": now.isoformat(),
                "request": request, "bounds": meta["bounds"], "model_id": meta["model_id"],
                "model_provider": meta.get("model_provider", "roboflow"),
                "weights_sha256": meta.get("weights_sha256"),
                "processing_profile": meta.get("processing_profile", "legacy"),
                "pipeline_version": meta.get("pipeline_version", "legacy"),
                "osm_timestamp": meta.get("osm_timestamp"), "imagery_acquisition_date": None,
                "osm_from_snapshot": meta.get("osm_from_snapshot", False),
                "osm_snapshot_sha256": meta.get("osm_snapshot_sha256"),
                "osm_layer_source": meta.get("osm_layer_source"), "osm_mvt_status": meta.get("osm_mvt_status"),
                "warnings": {"original_osm": meta["osm_source_error"]} if meta.get("osm_source_error") else {},
                "locked_inputs_from_run_id": meta.get("locked_inputs_from_run_id"),
                "inference_seconds": meta.get("inference_seconds"),
                "image_size": list(image.size) if image is not None else None,
                "tile_size": meta.get("tile_size"), "tile_overlap": meta.get("tile_overlap"), "tiles": meta.get("tiles", 0),
                "status": "partial" if errors else "complete", "errors": errors,
                "code_sha256": PROCESSING_CODE_SHA256, "artifacts": artifacts,
                "reference_status": "missing", "coverage_verified": False}
    # Manifest en son yazılır; böylece yarım kalan kayıtlar okunmaz.
    (directory / "manifest.tmp").write_bytes(_json_bytes(manifest))
    (directory / "manifest.tmp").replace(directory / "manifest.json")
    return manifest


def load_manifest(run_id, verify=True):
    if not re.fullmatch(r"\d{8}T\d{6}_[a-f0-9]{12}", run_id):
        raise HTTPException(422, "Geçersiz analiz kaydı kimliği.")
    directory = ROOT / run_id
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if verify:
            for name, entry in manifest["artifacts"].items():
                if name not in ARTIFACTS:
                    raise ValueError("unknown artifact")
                raw = (directory / name).read_bytes()
                if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                    raise ValueError("checksum mismatch")
    except FileNotFoundError:
        raise HTTPException(404, "Analiz kaydı veya bileşeni bulunamadı.")
    except (ValueError, KeyError):
        raise HTTPException(409, "Analiz arşivinin bütünlük kontrolü başarısız.")
    return manifest


@router.get("")
def list_runs():
    runs = []
    if ROOT.exists():
        for path in sorted(ROOT.glob("*/manifest.json"), reverse=True)[:100]:
            try:
                m = json.loads(path.read_text(encoding="utf-8"))
                runs.append({k: m[k] for k in ("id", "captured_at", "request", "status", "reference_status")})
            except (ValueError, KeyError):
                continue
    return {"runs": runs}


@router.get("/{run_id}")
def get_run(run_id: str):
    return load_manifest(run_id)


@router.get("/{run_id}/{artifact}")
def get_artifact(run_id: str, artifact: str):
    if artifact not in ARTIFACTS:
        raise HTTPException(404, "Bilinmeyen analiz bileşeni.")
    manifest = load_manifest(run_id)
    if artifact not in manifest["artifacts"]:
        raise HTTPException(404, "Bu analizde istenen bileşen yok.")
    return FileResponse(ROOT / run_id / artifact, media_type=ARTIFACTS[artifact])
