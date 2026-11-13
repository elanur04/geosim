"""Bartın için canlı veri kaynağı kontrolü: kaynaklardan gelen bina/yol sayılarını raporlar."""
import json
from pathlib import Path

from osm_source import fetch_osm_source

if __name__ == "__main__":
    # Bartın merkezi ve çevresindeki mahalleler.
    bounds = {"south": 41.626, "west": 32.324, "north": 41.642, "east": 32.348}
    result = fetch_osm_source(bounds)
    output = Path(__file__).parent / "verification"
    output.mkdir(exist_ok=True)
    summary = {"bounds": bounds, "buildings": len(result["buildings"]),
               "roads": len(result["roads"]), "error": result["error"],
               "osm_timestamp": result["timestamp"], "coverage_verified": False}
    (output / "bartin-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if result["error"]:
        print(json.dumps(summary, ensure_ascii=False))
        raise SystemExit(1)
    (output / "bartin-source.geojson").write_text(json.dumps({
        "type": "FeatureCollection", "features": result["buildings"] + result["roads"],
        "meta": summary}, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
