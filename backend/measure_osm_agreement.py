"""Arşivlenmiş model çıktılarının özgün OSM poligonlarını ne kadar bulduğunu ölçer.

Bu bir doğruluk (F1) ölçümü değildir: OSM eksik olabilir ve çatı-zemin kayması
içerir. Bu yüzden yalnızca OSM binalarının bulunma oranı raporlanır; OSM'de
olmayan tespitlere yanlış denemez. D1 ve test pencereleri burada kullanılmaz.
"""
import argparse
import json
from pathlib import Path

import analysis_archive
from analysis_archive import load_manifest
from evaluate_buildings import measure

DEFAULT_RUNS = ("20261104T182547_", "20261104T182507_")
OUTPUT = Path(__file__).parent / "verification" / "model-osm-agreement.json"


def agreement(run_id):
    manifest = load_manifest(run_id)
    if manifest["status"] != "complete" or manifest.get("osm_layer_source") != "original_osm":
        raise ValueError(f"{run_id}: tamamlanmış ve özgün OSM içeren analiz gerekir")
    directory = analysis_archive.ROOT / run_id
    osm = json.loads((directory / "osm.geojson").read_text(encoding="utf-8"))
    ai = json.loads((directory / "ai.geojson").read_text(encoding="utf-8"))
    original = [f for f in osm["features"] if f["properties"].get("type") == "building"
                and f["properties"].get("geometry_source") == "overpass"]
    reference = {"type": "FeatureCollection", "features": original}
    results = {}
    for threshold in (.5, .3):
        result = measure(ai, reference, manifest["bounds"], threshold)
        results[f"iou_{threshold}"] = {"found": result["tp"], "missed": result["fn"], "found_ratio": result["recall"]}
    return {"analysis_run_id": run_id, "model_id": manifest["model_id"], "confidence": manifest["request"]["confidence"],
            "original_osm_buildings": len(original), **results}


def resolve(prefix):
    matches = sorted(path.name for path in analysis_archive.ROOT.glob(f"{prefix}*") if path.is_dir())
    if len(matches) != 1:
        raise ValueError(f"{prefix}: tek bir analiz bulunamadı")
    return matches[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", default=DEFAULT_RUNS, help="analiz kimliği veya kimlik başlangıcı")
    args = parser.parse_args()
    report = {"note": "F1 değildir; özgün OSM binalarının bulunma oranıdır.",
              "runs": [agreement(resolve(run)) for run in args.runs]}
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
