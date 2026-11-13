import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

import analysis_archive
from evaluate_buildings import compare_methods, evaluate_run, measure, original_osm_methods

BOUNDS = {"south": -1, "west": -1, "north": 3, "east": 3}


def building(fid, x0=0, y0=0, x1=1, y1=1):
    return {"type": "Feature", "id": fid, "properties": {"type": "building", "source": "manual"},
            "geometry": {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}}


def collection(*features):
    return {"type": "FeatureCollection", "features": list(features), "meta": {"bounds": BOUNDS}}


class EvaluationTests(unittest.TestCase):
    def test_duplicate_predictions_are_false_positives_not_extra_true_positives(self):
        result = measure(collection(building('p1'), building('p2')), collection(building('r')), BOUNDS)
        self.assertEqual((result['tp'], result['fp'], result['fn']), (1, 1, 0))
        self.assertAlmostEqual(result['precision'], .5)
        self.assertAlmostEqual(result['f1'], 2 / 3)

    def test_matching_maximizes_number_of_valid_pairs_before_iou(self):
        truth = collection(building('a'), building('b', .3, 0, 1.3, 1))
        pred = collection(building('p1'), building('p2', -.3, 0, .7, 1))
        result = measure(pred, truth, BOUNDS)
        self.assertEqual(result['tp'], 2)
        self.assertEqual(result['fp'], 0)

    def test_invalid_prediction_counts_as_false_positive(self):
        invalid = building('broken')
        invalid['geometry']['coordinates'][0].pop()
        result = measure(collection(invalid), collection(building('r')), BOUNDS)
        self.assertEqual((result['tp'], result['fp'], result['fn']), (0, 1, 1))
        self.assertEqual(result['valid_closed_polygon_ratio'], 0)

    def test_mapbox_supplement_is_removed_from_osm_and_hybrid_arms(self):
        original = building('osm_way_1')
        original['properties'].update(source='osm', geometry_source='overpass')
        supplement = building('b_osm_1', 1.5, 0, 2.5, 1)
        supplement['properties'].update(source='osm')
        methods = {'osm': collection(original, supplement), 'ai': collection(), 'hybrid': collection(original, supplement)}
        manifest = {'osm_layer_source': 'original_osm', 'bounds': BOUNDS, 'request': {'confidence': .25}}
        cleaned, arm = original_osm_methods(methods, manifest)
        self.assertEqual(arm, 'original_osm_only')
        self.assertEqual([f['id'] for f in cleaned['osm']['features']], ['osm_way_1'])
        self.assertEqual([f['id'] for f in cleaned['hybrid']['features']], ['osm_way_1'])
        self.assertEqual(len(methods['osm']['features']), 2)

    def test_run_without_original_osm_is_marked_not_relabelled(self):
        methods = {'osm': collection(building('b_osm_1')), 'ai': collection(), 'hybrid': collection()}
        cleaned, arm = original_osm_methods(methods, {'osm_layer_source': 'mapbox_mvt'})
        self.assertEqual(arm, 'not_original_osm')
        self.assertIs(cleaned, methods)

    def test_paired_bootstrap_is_deterministic_and_not_a_coverage_claim(self):
        truth = collection(building('r1'), building('r2', 1.5, 0, 2.5, 1))
        methods = {'osm': collection(building('o1')), 'ai': collection(building('a1')), 'hybrid': copy.deepcopy(truth)}
        first = compare_methods(methods, truth, BOUNDS, repetitions=100)
        second = compare_methods(methods, truth, BOUNDS, repetitions=100)
        self.assertEqual(first, second)
        self.assertTrue(first['target_five_points_met'])
        self.assertAlmostEqual(first['hybrid_minus_osm_f1_pp'], 100 / 3)
        self.assertEqual(len(first['paired_ci95_pp']), 2)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root_patch = patch.object(analysis_archive, 'ROOT', Path(self.temp.name))
        self.root_patch.start()
        app = FastAPI()
        app.include_router(analysis_archive.router)
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close(); self.root_patch.stop(); self.temp.cleanup()

    def archive(self, image=True):
        payload = collection(building('hybrid'))
        payload['meta'].update(model_id='test/1', osm_timestamp='2026-11-01', tiles=4, osm_input_complete=True)
        return analysis_archive.archive_analysis(request={'confidence': .25}, payload=payload,
            image=Image.new('RGB', (16, 16)) if image else None, osm_features=[building('osm')],
            ai_features=[building('ai')], model_output={'predictions': []})

    def test_complete_archive_is_downloadable_without_network(self):
        manifest = self.archive()
        result = self.client.get(f"/api/analysis-runs/{manifest['id']}")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()['status'], 'complete')
        image = self.client.get(f"/api/analysis-runs/{manifest['id']}/image.png")
        self.assertEqual(image.status_code, 200)
        self.assertEqual(image.headers['content-type'], 'image/png')

    def test_corruption_is_detected_before_evaluation(self):
        manifest = self.archive()
        (analysis_archive.ROOT / manifest['id'] / 'osm.geojson').write_text('{}')
        result = self.client.get(f"/api/analysis-runs/{manifest['id']}")
        self.assertEqual(result.status_code, 409)

    def test_missing_image_makes_run_partial(self):
        self.assertEqual(self.archive(image=False)['status'], 'partial')

    def test_osm_or_ai_output_is_not_accepted_as_independent_reference(self):
        manifest = self.archive()
        reference = analysis_archive.ROOT / manifest['id'] / 'osm.geojson'
        with self.assertRaisesRegex(ValueError, 'Independent'):
            evaluate_run(manifest['id'], reference, repetitions=100)

    def test_matching_manual_reference_produces_report(self):
        manifest = self.archive()
        reference = collection(building('manual_ref'))
        reference['meta'].update(role='reference', annotation_method='manual_from_image', reference_complete=True,
                                 image_sha256=manifest['artifacts']['image.png']['sha256'])
        path = Path(self.temp.name) / 'reference.geojson'
        path.write_text(json.dumps(reference), encoding='utf-8')
        report = evaluate_run(manifest['id'], path, repetitions=100)
        self.assertEqual(report['methods']['hybrid']['f1'], 1)
        self.assertFalse(report['target_five_points_met'])
        self.assertEqual(report['paired_ci95_pp'], [0, 0])


if __name__ == '__main__':
    unittest.main()
