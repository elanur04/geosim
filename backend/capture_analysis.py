"""Sabit bir değerlendirme penceresinin analiz çıktısını arşive kaydeder."""
import argparse
import json
import time
from pathlib import Path

import requests


def capture(name, lat, lon, zoom=17, confidence=.25, api="http://127.0.0.1:8000"):
    print(f"Capturing {name}: ({lat}, {lon}), zoom={zoom}, confidence={confidence}", flush=True)
    t0 = time.perf_counter()
    response = requests.post(f"{api}/analyze", json={"latitude": lat, "longitude": lon,
        "zoom": zoom, "confidence": confidence, "include_ai": True}, timeout=(5, 180))
    response.raise_for_status()
    payload = response.json()
    meta = payload["meta"]
    run_id = meta.get("analysis_run_id")
    if not run_id:
        raise ValueError(f"No reproducible archive created: {meta.get('archive_error', 'unknown error')}")
    manifest_response = requests.get(f"{api}/api/analysis-runs/{run_id}", timeout=30)
    manifest_response.raise_for_status()
    manifest = manifest_response.json()
    summary = {"case": name, "analysis_run_id": run_id, "status": manifest["status"],
               "request": manifest["request"], "bounds": manifest["bounds"],
               "model_id": manifest["model_id"], "osm_buildings": meta["osm_vector_buildings"],
               "osm_layer_source": meta["osm_layer_source"], "osm_from_snapshot": meta["osm_from_snapshot"],
               "ai_buildings": meta["ai_raw_detections"], "hybrid_buildings": meta["detected_buildings"],
               "roads": sum(f["properties"].get("type") == "road" for f in payload["features"]),
               "tiles": meta["tiles"], "failed_tiles": meta["failed_tiles"],
               "request_seconds": round(time.perf_counter() - t0, 3),
               "reference_status": "missing", "f1": None, "five_point_target_verified": False,
               "errors": manifest["errors"]}
    summary["warnings"] = manifest.get("warnings", {})
    path = Path(__file__).parent / "verification" / f"{name}-analysis.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True, choices=['bartin-center', 'istanbul-balat', 'bartin-campus'])
    parser.add_argument('--lat', type=float, required=True)
    parser.add_argument('--lon', type=float, required=True)
    parser.add_argument('--zoom', type=float, default=17)
    parser.add_argument('--confidence', type=float, default=.25)
    args = parser.parse_args()
    summary = capture(args.name, args.lat, args.lon, args.zoom, args.confidence)
    raise SystemExit(0 if summary['status'] == 'complete' else 1)
