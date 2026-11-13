import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

import regions


def sample_collection():
    return {"type": "FeatureCollection", "meta": {"bounds": {"west": 32, "south": 41, "east": 33, "north": 42}},
        "features": [
            {"type": "Feature", "id": "building_1", "geometry": {"type": "Polygon", "coordinates": [
                [[32.1, 41.1], [32.2, 41.1], [32.2, 41.2], [32.1, 41.2], [32.1, 41.1]]]},
             "properties": {"type": "building", "height": 9, "height_source": "display_default", "review_status": "unreviewed"}},
            {"type": "Feature", "id": "road_1", "geometry": {"type": "LineString", "coordinates": [[32.1, 41.1], [32.3, 41.3]]},
             "properties": {"type": "road", "width": 3, "review_status": "unreviewed"}}]}


class RegionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db_patch = patch.object(regions, "DB_PATH", Path(self.temp.name) / "regions.sqlite3")
        self.db_patch.start()
        app = FastAPI()
        app.include_router(regions.router)
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        self.db_patch.stop()
        self.temp.cleanup()

    def save(self, collection=None, revision=0, **kwargs):
        return self.client.put('/api/regions/test', json={"name": "Test region", "revision": revision,
            "collection": sample_collection() if collection is None else collection, **kwargs})

    def test_bartin_snapshot_loads_without_external_services(self):
        response = self.client.get('/api/regions/bartin')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['quality']['buildings'], 511)
        self.assertEqual(data['quality']['roads'], 663)
        self.assertFalse(data['quality']['coverage_reviewed'])
        self.assertFalse(data['quality']['simulation_ready'])

    def test_save_reopen_and_deletion_history(self):
        self.assertEqual(self.save().status_code, 200)
        changed = sample_collection()
        changed['features'] = changed['features'][1:]
        self.assertEqual(self.save(changed, revision=1).json()['revision'], 2)
        self.assertEqual(len(self.client.get('/api/regions/test').json()['collection']['features']), 1)
        previous = self.client.get('/api/regions/test?revision=1').json()['collection']
        self.assertEqual(len(previous['features']), 2)
        self.assertEqual(self.save(previous, revision=2).json()['revision'], 3)
        self.assertEqual(len(self.client.get('/api/regions/test').json()['collection']['features']), 2)

    def test_stale_save_cannot_overwrite_corrections(self):
        self.assertEqual(self.save().status_code, 200)
        stale = self.save(revision=0)
        self.assertEqual(stale.status_code, 409)
        self.assertEqual(self.client.get('/api/regions/test').json()['revision'], 1)

    def test_self_intersecting_polygon_and_open_ring_are_rejected(self):
        for ring in (
            [[32.1, 41.1], [32.2, 41.2], [32.1, 41.2], [32.2, 41.1], [32.1, 41.1]],
            [[32.1, 41.1], [32.2, 41.1], [32.2, 41.2], [32.1, 41.2]],
        ):
            data = sample_collection()
            data['features'][0]['geometry']['coordinates'] = [ring]
            self.assertEqual(self.save(data).status_code, 422)

    def test_outside_geometry_and_duplicate_ids_are_rejected(self):
        data = sample_collection()
        data['features'][1]['geometry']['coordinates'] = [[28, 40], [29, 40]]
        self.assertEqual(self.save(data).status_code, 422)
        data = sample_collection()
        data['features'][1]['id'] = 'building_1'
        self.assertEqual(self.save(data).status_code, 422)

    def test_complete_review_is_not_claimed_to_prove_simulation_readiness(self):
        data = sample_collection()
        for f in data['features']:
            f['properties']['review_status'] = 'reviewed'
        result = self.save(data, coverage_reviewed=True).json()
        self.assertTrue(result['quality']['geometry_review_complete'])
        self.assertFalse(result['quality']['simulation_ready'])
        self.assertEqual(result['quality']['unknown_heights'], 1)

    def test_ui_properties_removed_without_losing_road_connectivity(self):
        data = sample_collection()
        props = data['features'][1]['properties']
        props.update(_idx=4, _fillColor=[0, 1, 2], osm_nodes=[1, 2], bridge='yes', layer='1')
        saved = self.save(data).json()['collection']['features'][1]['properties']
        self.assertNotIn('_idx', saved)
        self.assertEqual(saved['osm_nodes'], [1, 2])
        self.assertEqual(saved['layer'], '1')

    def test_bad_metadata_and_dimensions_return_validation_errors(self):
        data = sample_collection()
        data['meta'] = None
        self.assertEqual(self.save(data).status_code, 422)
        for value in (-3, 0, True, 'not a number'):
            data = sample_collection()
            data['features'][1]['properties']['width'] = value
            self.assertEqual(self.save(data).status_code, 422)


if __name__ == '__main__':
    unittest.main()
