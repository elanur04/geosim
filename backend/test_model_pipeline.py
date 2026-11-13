import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from shapely.geometry import Polygon
import analysis_archive
import api
import building_models
from fastapi.testclient import TestClient
from prediction_pipeline import deduplicate_predictions
from vectorize import regularize_pixel_polygon, roboflow_predictions_to_geojson, pixel_to_geo_bbox


def prediction(points, tile=None, score=.8):
    return {'points':[{'x':x,'y':y} for x,y in points], 'confidence':score,
            'tile':tile,'offset_applied':True}


class GeometryTests(unittest.TestCase):
    def test_l_shape_survives_regularization(self):
        points=[[0,0],[40,0],[40,10],[10,10],[10,40],[0,40]]
        original=Polygon(points)
        actual=Polygon(regularize_pixel_polygon(points))
        self.assertGreaterEqual(actual.intersection(original).area/actual.union(original).area,.98)
        self.assertGreater(len(actual.exterior.coords),5)

    def test_large_single_building_is_not_a_blob_and_duplicates_are_removed(self):
        large=prediction([[0,0],[90,0],[90,90],[0,90]],[0,0,640,640])
        duplicate={**copy.deepcopy(large),'tile':[10,10,650,650],'confidence':.7}
        self.assertEqual(len(deduplicate_predictions([large,duplicate])),1)
        self.assertEqual(Polygon([(p['x'],p['y']) for p in deduplicate_predictions([large])[0]['points']]).area,8100)

    def test_adjacent_buildings_with_overlapping_boxes_are_preserved(self):
        first=prediction([[0,0],[40,0],[40,10],[10,10],[10,40],[0,40]])
        second=prediction([[12,12],[38,12],[38,38],[12,38]])
        self.assertEqual(len(deduplicate_predictions([first,second])),2)

    def test_provider_provenance_unique_ids_and_no_road_clipping(self):
        preds=[prediction([[10,10],[30,10],[30,30],[10,30]]),prediction([[40,40],[60,40],[60,60],[40,60]])]
        result={'predictions':preds,'provider':'geoai','image':{'width':100,'height':100}}
        with patch('vectorize._build_road_corridor_union',side_effect=AssertionError('Must not alter AI using OSM')):
            features=roboflow_predictions_to_geojson(result,bounds={'west':0,'east':1,'south':0,'north':1},osm_roads=[])['features']
        self.assertEqual(len({f['id'] for f in features}),2)
        self.assertEqual({f['properties']['source'] for f in features},{'geoai'})

    def test_heuristic_score_is_not_ai_confidence(self):
        p=prediction([[10,10],[30,10],[30,30],[10,30]])
        p['prediction_source']='color_heuristic'
        feature=roboflow_predictions_to_geojson([p])['features'][0]
        self.assertIsNone(feature['properties']['ai_confidence'])
        self.assertEqual(feature['properties']['heuristic_score'],.8)

    def test_web_mercator_conversion_matches_reference_annotator(self):
        import math
        b={'west':0,'east':1,'south':40,'north':60}
        lon,lat=pixel_to_geo_bbox(50,50,b,100,100)
        y=(math.asinh(math.tan(math.radians(60)))+math.asinh(math.tan(math.radians(40))))/2
        self.assertAlmostEqual(lat,math.degrees(math.atan(math.sinh(y))),places=6)


class ModelIntegrationTests(unittest.TestCase):
    def test_missing_local_weights_never_fall_back_to_cloud(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(building_models,'weight_path',return_value=Path(tmp)/'missing.pth'):
            with self.assertRaisesRegex(ValueError,'Missing weights'):
                building_models.scan_buildings(Image.new('RGB',(16,16)),provider='geoai')

    def test_osm_only_does_not_download_imagery_or_invoke_model(self):
        def vectors(*args,**kwargs):
            return [],[],{'complete':True,'tiles':1,'failed_tiles':0}
        with patch.object(api,'MAPBOX_ACCESS_TOKEN','test'), patch.object(api,'_ANALYZE_CACHE',{}), \
             patch.object(api,'fetch_mapbox_osm_vectors',side_effect=vectors) as vector_mock, \
             patch.object(api,'_get_async_client',side_effect=AssertionError('Image download forbidden')), \
             patch.object(api,'scan_buildings',side_effect=AssertionError('Inference forbidden')), TestClient(api.app) as client:
            vector_mock.__name__='vectors'
            response=client.post('/analyze',json={'latitude':41.6374,'longitude':32.3381,'zoom':17,'include_ai':False})
            self.assertEqual(response.status_code,200)
            self.assertEqual(response.json()['meta']['ai_processed_detections'],0)


if __name__=='__main__':
    unittest.main()
