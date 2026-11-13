"""Kayıtlı deneylerin değerlendirme girdilerini denetler; referansı olmayan deney için F1 hesaplanmaz."""
import hashlib
import json
from pathlib import Path

from PIL import Image
from shapely.geometry import shape

import analysis_archive
from analysis_archive import load_manifest
from capture_thresholds import CASES, VERIFICATION


def digest_features(features):
    return hashlib.sha256(json.dumps(features, sort_keys=True).encode()).hexdigest()


def audit():
    cases = []
    for case in CASES:
        protocol = json.loads((VERIFICATION / f'{case}-thresholds.json').read_text(encoding='utf-8'))
        baseline = load_manifest(protocol['baseline_run_id'])
        base_dir = analysis_archive.ROOT / baseline['id']
        baseline_osm = json.loads((base_dir / 'osm.geojson').read_text(encoding='utf-8'))['features']
        records = []
        for entry in protocol['runs']:
            manifest = load_manifest(entry['analysis_run_id'])
            directory = analysis_archive.ROOT / manifest['id']
            assert manifest['status'] == 'complete'
            assert manifest['artifacts']['image.png']['sha256'] == protocol['fixed_image_sha256']
            with Image.open(directory / 'image.png') as image:
                assert list(image.size) == manifest['image_size']
            model = json.loads((directory / 'model-output.json').read_text(encoding='utf-8'))
            assert len(model['raw_tiles']) == manifest['tiles']
            assert all(isinstance(t['response'].get('predictions'), list) for t in model['raw_tiles'])
            stats = {}
            for method in ('osm', 'ai', 'hybrid'):
                data = json.loads((directory / f'{method}.geojson').read_text(encoding='utf-8'))
                assert data['meta']['bounds'] == manifest['bounds']
                if method == 'osm':
                    assert digest_features(data['features']) == digest_features(baseline_osm)
                buildings = [f for f in data['features'] if f['properties']['type'] == 'building']
                valid = 0
                for f in buildings:
                    raw = f['geometry']
                    polys = [raw['coordinates']] if raw['type'] == 'Polygon' else raw['coordinates']
                    closed = all(len(r) >= 4 and r[0] == r[-1] for p in polys for r in p)
                    geom = shape(raw)
                    valid += bool(closed and geom.is_valid and not geom.is_empty)
                stats[method] = {'buildings': len(buildings), 'valid_closed_buildings': valid,
                                 'valid_closed_ratio': valid / len(buildings) if buildings else None}
            records.append({'run_id': manifest['id'], 'threshold': manifest['request']['confidence'],
                            'tiles': manifest['tiles'], 'input_integrity_verified': True, 'geometry': stats})
        assert {r['threshold'] for r in records} == {.25, .4, .6}
        cases.append({'case': case, 'runs': records, 'reference_status': 'not_supplied_to_this_audit'})
    output = {'cases': cases, 'complete_experiments': sum(len(c['runs']) for c in cases),
              'input_integrity_verified': True, 'f1_target_verified': False,
              'note': 'Polygon validity and data integrity do not establish detection precision or recall.'}
    (VERIFICATION / 'evaluation-input-audit.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps({'complete_experiments': output['complete_experiments'], 'input_integrity_verified': True,
                      'f1_target_verified': False}))


if __name__ == '__main__':
    audit()
