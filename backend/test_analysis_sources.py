import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import api
import osm_source
import vectorize


class AnalysisSourceTests(unittest.TestCase):
    def test_nearby_centers_do_not_share_rounded_response(self):
        def vectors(bounds, token, include_status=False):
            return [], [], {"complete": True, "tiles": 1, "failed_tiles": 0}
        with patch.object(api, 'MAPBOX_ACCESS_TOKEN', 'test'), patch.object(api, '_ANALYZE_CACHE', {}), \
                patch.object(api, 'fetch_mapbox_osm_vectors', side_effect=vectors) as fetch, TestClient(api.app) as client:
            fetch.__name__ = 'fetch_mapbox_osm_vectors'
            # Açılıştaki ön ısınma kapatılır; yanıt anahtarlarını etkilemez.
            fetch.reset_mock()
            first = client.post('/analyze', json={'latitude': 41.6374, 'longitude': 32.3381, 'zoom': 17, 'include_ai': False}).json()
            second = client.post('/analyze', json={'latitude': 41.63741, 'longitude': 32.3381, 'zoom': 17, 'include_ai': False}).json()
            self.assertNotEqual(first['meta']['bounds']['north'], second['meta']['bounds']['north'])
            self.assertEqual(second['meta']['center']['lat'], 41.63741)
            self.assertTrue(first['meta']['osm_mvt_status']['complete'])

    def test_missing_vector_tiles_are_reported_and_not_cached(self):
        bounds = {'south': 41.635, 'west': 32.335, 'north': 41.638, 'east': 32.339}
        def missing_tiles(fn, jobs):
            return [(x, y, None) for _, x, y, _ in jobs]
        with patch.object(vectorize, '_VECTOR_BBOX_CACHE', {}), patch.object(vectorize._SHARED_POOL, 'map', side_effect=missing_tiles):
            buildings, roads, status = vectorize.fetch_mapbox_osm_vectors(bounds, 'test', include_status=True)
            self.assertFalse(status['complete'])
            self.assertGreater(status['failed_tiles'], 0)
            self.assertEqual(vectorize._VECTOR_BBOX_CACHE, {})
            self.assertEqual((buildings, roads), ([], []))

    def test_original_bartin_snapshot_does_not_require_network(self):
        bounds = {'south': 41.635, 'west': 32.335, 'north': 41.638, 'east': 32.339}
        with patch.object(osm_source, '_CACHE', {}), patch.object(osm_source.requests, 'get') as get:
            result = osm_source.fetch_osm_source(bounds)
        get.assert_not_called()
        self.assertTrue(result['from_snapshot'])
        self.assertIsNone(result['error'])
        self.assertGreater(len(result['buildings']), 0)
        self.assertGreater(len(result['roads']), 0)


if __name__ == '__main__':
    unittest.main()
