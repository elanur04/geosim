"""Aynı kayıtlı görüntü ve OSM verisi üzerinde modeli .40/.60 eşikleriyle ayrı ayrı çalıştırır.

.25 çıktısını sonradan filtrelemek modeli o eşikle çalıştırmakla aynı değildir;
bu yüzden her eşik için model yeniden çağrılır ve kaynak bilgisi kaydedilir.
"""
import argparse
import copy
import json
import time
from pathlib import Path

from PIL import Image

import analysis_archive
from analysis_archive import archive_analysis, load_manifest
from vectorize import (scan_satellite_image_roboflow, roboflow_predictions_to_geojson,
                       merge_hybrid_buildings, MAIN_IMAGE_W)

CASES = ('bartin-center', 'bartin-campus', 'istanbul-balat')
VERIFICATION = Path(__file__).parent / 'verification'


def capture_case(case, thresholds):
    summary = json.loads((VERIFICATION / f'{case}-analysis.json').read_text(encoding='utf-8'))
    baseline = load_manifest(summary['analysis_run_id'])
    if baseline['status'] != 'complete':
        raise ValueError(f'{case}: baseline inputs are incomplete')
    directory = analysis_archive.ROOT / baseline['id']
    image = Image.open(directory / 'image.png').convert('RGB')
    osm_features = json.loads((directory / 'osm.geojson').read_text(encoding='utf-8'))['features']
    records = [{'threshold': baseline['request']['confidence'], 'analysis_run_id': baseline['id'], 'status': 'complete'}]
    for confidence in thresholds:
        if confidence == baseline['request']['confidence']:
            continue
        print(f'{case}: real inference at confidence={confidence}, fixed image {baseline["id"]}', flush=True)
        t0 = time.perf_counter()
        result = scan_satellite_image_roboflow(image, model_id=baseline['model_id'], confidence_threshold=confidence,
            tile_size=baseline['tile_size'], overlap=baseline['tile_overlap'], target_size=image.size)
        elapsed = time.perf_counter() - t0
        ai = roboflow_predictions_to_geojson(result, bounds=baseline['bounds'], img_w=image.width, img_h=image.height,
                                             simplify_epsilon=1.5 * image.width / MAIN_IMAGE_W)['features']
        buildings = [copy.deepcopy(f) for f in osm_features if f['properties']['type'] == 'building']
        roads = [copy.deepcopy(f) for f in osm_features if f['properties']['type'] == 'road']
        hybrid = merge_hybrid_buildings(buildings, ai)
        payload = {'type': 'FeatureCollection', 'features': hybrid + roads, 'meta': {
            'bounds': baseline['bounds'], 'model_id': baseline['model_id'], 'osm_input_complete': True,
            'osm_layer_source': baseline['osm_layer_source'], 'osm_from_snapshot': True,
            'osm_snapshot_sha256': baseline['artifacts']['osm.geojson']['sha256'],
            'osm_timestamp': baseline['osm_timestamp'], 'osm_mvt_status': baseline.get('osm_mvt_status'),
            'tiles': result['tiles'], 'failed_tiles': result['failed_tiles'],
            'tile_size': baseline['tile_size'], 'tile_overlap': baseline['tile_overlap'],
            'locked_inputs_from_run_id': baseline['id'], 'inference_seconds': round(elapsed, 3)}}
        request = {**baseline['request'], 'confidence': confidence}
        manifest = archive_analysis(request=request, payload=payload, image=image, osm_features=osm_features,
                                    ai_features=ai, model_output=result)
        assert manifest['artifacts']['image.png']['sha256'] == baseline['artifacts']['image.png']['sha256']
        records.append({'threshold': confidence, 'analysis_run_id': manifest['id'], 'status': manifest['status'],
                        'ai_buildings': len(ai), 'hybrid_buildings': len(hybrid), 'inference_seconds': round(elapsed, 3),
                        'failed_tiles': result['failed_tiles'], 'f1': None})
        print(json.dumps(records[-1]), flush=True)
        if manifest['status'] != 'complete':
            break
    report = {'case': case, 'baseline_run_id': baseline['id'], 'fixed_image_sha256': baseline['artifacts']['image.png']['sha256'],
              'runs': records, 'best_threshold': None, 'selection_status': 'awaiting_independent_manual_reference'}
    (VERIFICATION / f'{case}-thresholds.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True, choices=[*CASES, 'all'])
    parser.add_argument('--thresholds', type=float, nargs='+', default=[.4, .6])
    args = parser.parse_args()
    if any(not 0 < value <= 1 for value in args.thresholds):
        parser.error('Thresholds must be in (0, 1]')
    for case in CASES if args.case == 'all' else [args.case]:
        report = capture_case(case, args.thresholds)
        if any(r['status'] != 'complete' for r in report['runs']):
            raise SystemExit('Stopped after incomplete inference; no further model calls made.')
