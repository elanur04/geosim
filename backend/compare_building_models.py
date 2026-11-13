"""Aday modelleri arşivlenmiş aynı görüntü + OSM verisi üzerinde karşılaştırır.

Bağımsız ve eksiksiz bir referans olmadan hiçbir model üstün sayılmaz.
--replay ile model çağrılmadan, çevrimdışı Roboflow işlem hattı karşılaştırması yapılır.
"""
import argparse
import copy
import json
import time
from pathlib import Path
from PIL import Image
import analysis_archive
from analysis_archive import load_manifest, archive_analysis
from building_models import scan_buildings, model_info
from prediction_pipeline import PIPELINE_VERSION, deduplicate_predictions
from vectorize import roboflow_predictions_to_geojson, merge_hybrid_buildings


def replay_tiles(output, image_size, confidence):
    """Eski yanıtlardaki normalize maskeleri şekil filtrelemesinden önceki haliyle geri alır."""
    width, height = image_size
    predictions = []
    for tile in output.get('raw_tiles', []):
        x0, y0, x1, y1 = tile['crop_box']
        response = tile.get('response')
        if not response or not isinstance(response.get('predictions'), list):
            raise ValueError('Replay requires successful raw responses for every tile.')
        size = response.get('image') or {}
        sx = (x1-x0) / (size.get('width') or (x1-x0))
        sy = (y1-y0) / (size.get('height') or (y1-y0))
        for index, pred in enumerate(response['predictions']):
            if float(pred.get('confidence', 0)) < confidence:
                continue
            points = [{'x': min(width, max(0, float(p['x'])*sx+x0)),
                       'y': min(height, max(0, float(p['y'])*sy+y0))} for p in pred.get('points', [])]
            if len(points) < 3:
                continue
            xs, ys = [p['x'] for p in points], [p['y'] for p in points]
            predictions.append({**pred, 'points': points, 'offset_applied': True,
                'tile': [x0,y0,x1,y1], 'width': max(xs)-min(xs), 'height': max(ys)-min(ys),
                'detection_id': f'replay_{x0}_{y0}_{index}',
                '_cut': bool((min(xs)<=x0+3 and x0>0) or (min(ys)<=y0+3 and y0>0)
                         or (max(xs)>=x1-3 and x1<width) or (max(ys)>=y1-3 and y1<height))})
    if not output.get('raw_tiles'):
        raise ValueError('No raw tiles in archive; cannot recreate unfiltered model output.')
    return {'predictions': deduplicate_predictions(predictions), 'raw_predictions': predictions,
            'raw_tiles': output['raw_tiles'], 'image': {'width': width, 'height': height},
            'tiles': len(output['raw_tiles']), 'failed_tiles': 0, 'provider': 'roboflow',
            'processing_profile': 'conservative', 'pipeline_version': PIPELINE_VERSION,
            'inference_mode': 'archived_response_replay'}


def run_comparison(run_id, providers, thresholds, replay=False):
    baseline = load_manifest(run_id)
    if baseline['status'] != 'complete':
        raise ValueError('Complete source archive required')
    directory = analysis_archive.ROOT / run_id
    image = Image.open(directory/'image.png').convert('RGB')
    osm = json.loads((directory/'osm.geojson').read_text(encoding='utf-8'))['features']
    records = []
    for provider in providers:
        for confidence in thresholds:
            started = time.perf_counter()
            if replay:
                if provider != 'roboflow' or confidence != baseline['request']['confidence']:
                    raise ValueError('Replay requires Roboflow and the ORIGINAL threshold; it is not a fresh inference run.')
                output = replay_tiles(json.loads((directory/'model-output.json').read_text(encoding='utf-8')), image.size, confidence)
                info = {'provider': provider, 'model_id': baseline['model_id'], 'weights_sha256': None}
            else:
                info = model_info(provider)
                output = scan_buildings(image, provider=provider, confidence_threshold=confidence,
                    tile_size=baseline.get('tile_size') or 640, overlap=baseline.get('tile_overlap') or 0,
                    target_size=image.size)
            elapsed = time.perf_counter()-started
            ai = roboflow_predictions_to_geojson(output, bounds=baseline['bounds'])['features']
            buildings = [copy.deepcopy(f) for f in osm if f['properties']['type']=='building']
            roads = [copy.deepcopy(f) for f in osm if f['properties']['type']=='road']
            hybrid = merge_hybrid_buildings(buildings, copy.deepcopy(ai), min_gapfill_conf=confidence,
                                            osm_roads=roads, bounds=baseline['bounds'])
            meta = {**baseline, 'model_id': info['model_id'], 'model_provider': provider,
                    'weights_sha256': info.get('weights_sha256'), 'processing_profile': 'conservative',
                    'pipeline_version': PIPELINE_VERSION, 'osm_input_complete': True,
                    'locked_inputs_from_run_id': run_id, 'tiles': output['tiles'],
                    'failed_tiles': output['failed_tiles'], 'inference_seconds': elapsed,
                    'tile_size': baseline.get('tile_size') or 640, 'tile_overlap': baseline.get('tile_overlap') or 0}
            request = {**baseline['request'], 'confidence': confidence, 'model_provider': provider,
                       'processing_profile': 'conservative', 'inference_mode': 'replay' if replay else 'model_call'}
            manifest = archive_analysis(request=request, payload={'type':'FeatureCollection', 'features':hybrid+roads, 'meta':meta},
                image=image, osm_features=osm, ai_features=ai, model_output=output)
            record = {'provider': provider, 'confidence': confidence, 'run_id': manifest['id'],
                      'status': manifest['status'], 'ai_count': len(ai), 'seconds': round(elapsed,3),
                      'mode': request['inference_mode'], 'f1': None}
            records.append(record)
            print(json.dumps(record), flush=True)
            if manifest['status'] != 'complete':
                raise ValueError('Partial inference; stopped, no accuracy claim.')
    return {'source_run_id': run_id, 'runs': records, 'winner': None,
            'selection_status': 'requires_independent_development_reference'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id')
    parser.add_argument('--providers', nargs='+', choices=['roboflow','geoai','yolo'], default=['geoai'])
    parser.add_argument('--thresholds', nargs='+', type=float, default=[.25])
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if any(not 0 < v <= 1 for v in args.thresholds):
        parser.error('Confidence must be in (0,1]')
    result = run_comparison(args.run_id, args.providers, args.thresholds, args.replay)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
