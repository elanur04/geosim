"""Bağımsız olarak etiketlenmiş, eksiksiz bir görüntüye karşı çevrimdışı bina değerlendirmesi.

Tahminler IoU >= 0.5 eşiğinde bire bir eşleştirilir. Eşli bootstrap birimleri referans
binalardır; eşleşmeyen tahminler en yakın referans merkezine atanır. Böylece yeniden
örneklemede hem yanlış pozitifler hem de kaçırılan binalar yer alır.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import box, shape
from shapely.strtree import STRtree

from analysis_archive import load_manifest
import analysis_archive


def _shapes(collection, bounds, reference=False):
    extent = box(bounds["west"], bounds["south"], bounds["east"], bounds["north"])
    shapes, ids, invalid, excluded = [], [], 0, 0
    for index, feature in enumerate(collection["features"]):
        if feature.get("properties", {}).get("type") != "building":
            continue
        try:
            raw = feature["geometry"]
            polys = [raw["coordinates"]] if raw["type"] == "Polygon" else raw["coordinates"]
            if raw["type"] not in ("Polygon", "MultiPolygon") or any(
                    len(ring) < 4 or ring[0] != ring[-1] for polygon in polys for ring in polygon):
                raise ValueError("not a closed polygon")
            geom = shape(raw)
            if not geom.is_valid or geom.is_empty:
                raise ValueError("invalid polygon")
            geom = geom.intersection(extent)
            if geom.is_empty or geom.area <= 0:
                excluded += 1
                continue
            shapes.append(geom)
            ids.append(str(feature.get("id", index)))
        except (ValueError, KeyError, TypeError):
            invalid += 1
    if reference and (invalid or excluded):
        raise ValueError("Reference contains invalid or out-of-frame buildings")
    return shapes, ids, invalid, excluded


def measure(prediction, reference, bounds, iou_threshold=0.5):
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")
    pred, pred_ids, invalid, outside = _shapes(prediction, bounds)
    truth, truth_ids, _, _ = _shapes(reference, bounds, reference=True)
    scores = np.zeros((len(pred), len(truth)), dtype=float)
    if truth:
        tree = STRtree(truth)
        for i, geom in enumerate(pred):
            for j in tree.query(geom):
                j = int(j)
                intersection = geom.intersection(truth[j]).area
                union = geom.area + truth[j].area - intersection
                scores[i, j] = intersection / union if union else 0
    accepted = scores + 1e-12 >= iou_threshold
    # Eşleşme sayısı önceliği, tek bir yüksek IoU çiftinin iki geçerli çifti engellemesini önler.
    weights = np.where(accepted, min(len(pred), len(truth)) + 1 + scores, 0)
    rows, cols = linear_sum_assignment(weights, maximize=True)
    matches = [(int(i), int(j), float(scores[i, j])) for i, j in zip(rows, cols) if accepted[i, j]]
    tp, fp, fn = len(matches), len(pred) + invalid - len(matches), len(truth) - len(matches)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    matched_pred = {i for i, _, _ in matches}
    matched_truth = {j for _, j, _ in matches}
    hits = np.array([int(j in matched_truth) for j in range(len(truth))], dtype=int)
    false_positives = np.zeros(len(truth), dtype=int)
    if truth:
        lat_scale = math.cos(math.radians((bounds["north"] + bounds["south"]) / 2))
        centers = np.array([[g.centroid.x * lat_scale, g.centroid.y] for g in truth])
        for i, geom in enumerate(pred):
            if i not in matched_pred:
                center = np.array([geom.centroid.x * lat_scale, geom.centroid.y])
                false_positives[int(np.argmin(np.sum((centers - center) ** 2, axis=1)))] += 1
    total = len(pred) + invalid
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1,
            "mean_matched_iou": float(np.mean([v for _, _, v in matches])) if matches else None,
            "invalid_polygons": invalid, "valid_closed_polygon_ratio": len(pred) / total if total else None,
            "excluded_outside_frame": outside,
            "matches": [{"prediction": pred_ids[i], "reference": truth_ids[j], "iou": v} for i, j, v in matches],
            "missed_reference_ids": [truth_ids[j] for j in range(len(truth)) if j not in matched_truth],
            "false_positive_ids": [pred_ids[i] for i in range(len(pred)) if i not in matched_pred],
            "_hits": hits, "_fp_groups": false_positives}


def compare_methods(methods, reference, bounds, repetitions=2000, seed=2026, iou_threshold=.5):
    if set(methods) != {"osm", "ai", "hybrid"}:
        raise ValueError("Exactly osm, ai and hybrid methods are required")
    results = {name: measure(data, reference, bounds, iou_threshold) for name, data in methods.items()}
    n = len(results["osm"]["_hits"])
    if not n:
        raise ValueError("A complete reference must contain at least one building")
    if any(r["invalid_polygons"] for r in results.values()):
        intervals = {name: None for name in ('osm', 'ai')}
        interval_note = "Unavailable: invalid predictions have no reliable spatial bootstrap group."
    else:
        rng = np.random.default_rng(seed)
        differences = {name: [] for name in ('osm', 'ai')}
        for _ in range(repetitions):
            sample = rng.integers(0, n, n)
            f1 = {}
            for name in ("osm", "ai", "hybrid"):
                r = results[name]
                tp = int(r["_hits"][sample].sum())
                fp = int(r["_fp_groups"][sample].sum())
                f1[name] = 2 * tp / (tp + fp + n)
            for name in differences:
                differences[name].append(100 * (f1["hybrid"] - f1[name]))
        intervals = {name: [float(v) for v in np.quantile(values, [0.025, 0.975])]
                     for name, values in differences.items()}
        interval_note = "Paired reference-building bootstrap; false positives grouped by nearest reference centroid."
    delta = 100 * (results["hybrid"]["f1"] - results["osm"]["f1"])
    for result in results.values():
        result.pop("_hits"); result.pop("_fp_groups")
    delta_ai = 100 * (results['hybrid']['f1'] - results['ai']['f1'])
    return {"iou_threshold": iou_threshold, "boundary_rule": "clip_to_fixed_image_frame", "methods": results,
            "hybrid_minus_osm_f1_pp": delta, "target_five_points_met": delta >= 5 - 1e-10,
            "hybrid_minus_ai_f1_pp": delta_ai,
            "target_five_points_over_both_met": min(delta, delta_ai) >= 5 - 1e-10,
            "hybrid_better_than_both": all(results["hybrid"]["f1"] > results[name]["f1"] for name in ("osm", "ai")),
            "paired_ci95_pp": intervals['osm'], "paired_ci95_hybrid_minus_ai_pp": intervals['ai'],
            "h1_supported": all(ci is not None and ci[0] > 0 for ci in intervals.values()),
            "bootstrap_repetitions": repetitions, "bootstrap_seed": seed,
            "bootstrap_note": interval_note}


def original_osm_methods(methods, manifest):
    """Değerlendirmede OSM ve hibrit kollarını yalnızca özgün OSM poligonlarıyla kurar.

    Canlı analiz, Overpass binalarını Mapbox vektör binalarıyla tamamlayıp osm.geojson
    dosyasına birlikte yazar. Bu tamamlama "yalnızca OSM" kolunu şişirmesin diye ayıklanır;
    hibrit de arşivdeki AI çıktısıyla aynı kuralla yeniden birleştirilir.
    """
    if manifest.get("osm_layer_source") != "original_osm":
        return methods, "not_original_osm"
    from vectorize import merge_hybrid_buildings
    features = methods["osm"]["features"]
    buildings = [f for f in features if f.get("properties", {}).get("type") == "building"
                 and f["properties"].get("geometry_source") == "overpass"]
    roads = [f for f in features if f.get("properties", {}).get("type") == "road"]
    ai = [f for f in methods["ai"]["features"] if f.get("properties", {}).get("type") == "building"]
    merged = merge_hybrid_buildings(copy.deepcopy(buildings), copy.deepcopy(ai),
                                    min_gapfill_conf=manifest["request"]["confidence"],
                                    osm_roads=copy.deepcopy(roads), bounds=manifest["bounds"],
                                    processing_profile=manifest.get("processing_profile") or "conservative")
    return {**methods, "osm": {**methods["osm"], "features": buildings + roads},
            "hybrid": {**methods["hybrid"], "features": merged + roads}}, "original_osm_only"


def _evaluate_bytes(run_id, raw_reference, repetitions=2000):
    manifest = load_manifest(run_id)
    if manifest["status"] != "complete":
        raise ValueError("Partial analysis runs cannot be used for accuracy claims")
    reference = json.loads(raw_reference)
    meta = reference.get("meta", {})
    image_hash = manifest["artifacts"]["image.png"]["sha256"]
    if (meta.get("role") != "reference" or meta.get("annotation_method") != "manual_from_image"
            or meta.get("reference_complete") is not True or meta.get("image_sha256") != image_hash
            or meta.get("bounds") != manifest["bounds"]):
        raise ValueError("Independent, complete manual reference for this exact image and bounds is required")
    if any(f.get("properties", {}).get("source") != "manual" for f in reference["features"]):
        raise ValueError("OSM/AI predictions cannot be used as manual reference")
    directory = analysis_archive.ROOT / run_id
    methods = {name: json.loads((directory / f"{name}.geojson").read_text(encoding="utf-8")) for name in ("osm", "ai", "hybrid")}
    methods, osm_arm = original_osm_methods(methods, manifest)
    report = compare_methods(methods, reference, manifest["bounds"], repetitions)
    report['sensitivity_iou_030'] = compare_methods(methods, reference, manifest['bounds'], repetitions, iou_threshold=.3)
    return {"analysis_run_id": run_id, "image_sha256": image_hash, "osm_arm": osm_arm,
            "reference_sha256": hashlib.sha256(raw_reference).hexdigest(),
            "second_reviewer_fraction": meta.get("second_reviewer_fraction", 0),
            "second_reviewer_area_fraction": meta.get("second_reviewer_area_fraction", 0),
            "second_reviewer": meta.get("second_reviewer"),
            "reviewed_at": meta.get("reviewed_at"),
            "inference_confidence": manifest["request"]["confidence"], **report}


def evaluate_reference(run_id, reference, repetitions=2000):
    return _evaluate_bytes(run_id, json.dumps(reference, ensure_ascii=False, allow_nan=False).encode('utf-8'), repetitions)


def evaluate_run(run_id, reference_path, repetitions=2000):
    return _evaluate_bytes(run_id, Path(reference_path).read_bytes(), repetitions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2000)
    args = parser.parse_args()
    if args.repetitions < 100:
        parser.error("At least 100 bootstrap repetitions required")
    try:
        result = evaluate_run(args.run_id, args.reference, args.repetitions)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Evaluation refused: {error}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("analysis_run_id", "hybrid_minus_osm_f1_pp", "target_five_points_met", "paired_ci95_pp")}))
