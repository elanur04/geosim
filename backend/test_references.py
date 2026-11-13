import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

import analysis_archive
import references


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.patches = [patch.object(analysis_archive, 'ROOT', Path(self.temp.name) / 'runs'),
                        patch.object(references, 'DB_PATH', Path(self.temp.name) / 'references.sqlite3')]
        for p in self.patches:
            p.start()
        bounds = {'south': 0, 'west': 0, 'north': 1, 'east': 1}
        self.feature = {'type': 'Feature', 'id': 'manual_ref', 'properties': {'type': 'building', 'source': 'manual'},
                        'geometry': {'type': 'Polygon', 'coordinates': [[[.1, .1], [.2, .1], [.2, .2], [.1, .2], [.1, .1]]]}}
        payload = {'type': 'FeatureCollection', 'features': [self.feature], 'meta': {'bounds': bounds,
                   'model_id': 'test/1', 'tiles': 1, 'osm_input_complete': True}}
        manifest = analysis_archive.archive_analysis(request={'confidence': .25}, payload=payload, image=Image.new('RGB', (32, 32)),
            osm_features=[self.feature], ai_features=[self.feature], model_output={})
        self.run_id = manifest['id']
        self.data = {'type': 'FeatureCollection', 'features': [self.feature], 'meta': {'bounds': bounds,
                    'role': 'reference', 'annotation_method': 'manual_from_image', 'reference_complete': False, 'dataset_role': 'development',
                    'image_sha256': manifest['artifacts']['image.png']['sha256']}}
        app = FastAPI()
        app.include_router(references.router)
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        for p in reversed(self.patches):
            p.stop()
        self.temp.cleanup()

    def put(self, data=None, revision=0):
        return self.client.put(f'/api/references/{self.run_id}', json={'revision': revision, 'collection': data or self.data})

    def test_draft_persists_but_cannot_be_used_to_claim_recall(self):
        self.assertEqual(self.put().status_code, 200)
        loaded = self.client.get(f'/api/references/{self.run_id}').json()
        self.assertEqual(loaded['revision'], 1)
        measured = self.client.post(f'/api/references/{self.run_id}/evaluate')
        self.assertEqual(measured.status_code, 422)

    def test_wrong_image_and_copied_predictions_rejected(self):
        data = copy.deepcopy(self.data)
        data['meta']['image_sha256'] = 'different-image'
        self.assertEqual(self.put(data).status_code, 422)
        data = copy.deepcopy(self.data)
        data['features'][0]['properties']['source'] = 'osm'
        self.assertEqual(self.put(data).status_code, 422)

    def test_complete_reference_enables_measurement_without_fabricated_advisor_approval(self):
        data = copy.deepcopy(self.data)
        data['meta']['reference_complete'] = True
        self.assertEqual(self.put(data).status_code, 200)
        measured = self.client.post(f'/api/references/{self.run_id}/evaluate')
        self.assertEqual(measured.status_code, 200)
        report = measured.json()
        self.assertEqual(report['reference_revision'], 1)
        self.assertEqual(report['methods']['hybrid']['f1'], 1)
        self.assertFalse(report['target_five_points_met'])
        self.assertFalse(report['advisor_review_requirement_met'])

    def test_concurrent_save_cannot_overwrite_manual_reference(self):
        self.assertEqual(self.put().status_code, 200)
        self.assertEqual(self.put(revision=0).status_code, 409)

    def test_empty_draft_can_save_but_not_be_marked_complete(self):
        data = copy.deepcopy(self.data)
        data['features'] = []
        self.assertEqual(self.put(data).status_code, 200)
        data['meta']['reference_complete'] = True
        self.assertEqual(self.put(data, revision=1).status_code, 422)

    def variant(self, confidence):
        baseline = analysis_archive.load_manifest(self.run_id)
        payload = {'type': 'FeatureCollection', 'features': [self.feature], 'meta': {
            'bounds': baseline['bounds'], 'model_id': 'test/1', 'tiles': 1, 'osm_input_complete': True,
            'locked_inputs_from_run_id': self.run_id}}
        return analysis_archive.archive_analysis(request={'confidence': confidence}, payload=payload, image=Image.new('RGB', (32, 32)),
            osm_features=[self.feature], ai_features=[self.feature], model_output={})

    def test_three_thresholds_share_one_manual_reference(self):
        data = copy.deepcopy(self.data)
        data['meta']['reference_complete'] = True
        self.assertEqual(self.put(data).status_code, 200)
        variant = self.variant(.4)
        self.variant(.6)
        loaded = self.client.get(f"/api/references/{variant['id']}")
        self.assertEqual(loaded.json()['analysis_run_id'], self.run_id)
        report = self.client.post(f"/api/references/{variant['id']}/compare-thresholds")
        self.assertEqual(report.status_code, 200)
        result = report.json()
        self.assertEqual(len(result['results']), 3)
        self.assertEqual(len({r['reference_sha256'] for r in result['results']}), 1)
        self.assertEqual(result['best_thresholds_for_this_image'], [.25, .4, .6])
        self.assertIsNone(result['selected_deployment_threshold'])

    def test_missing_threshold_does_not_produce_a_best_threshold_claim(self):
        self.put()
        self.variant(.4)
        result = self.client.post(f'/api/references/{self.run_id}/compare-thresholds')
        self.assertEqual(result.status_code, 422)


if __name__ == '__main__':
    unittest.main()
