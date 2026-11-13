import unittest
from unittest.mock import patch

from shapely.geometry import shape

from osm_source import fetch_osm_source, parse_osm, supplement_buildings, _CACHE


BOUNDS = {"south": 0, "west": 0, "north": 5, "east": 5}


def ring(x0, y0, x1, y1):
    return [{"lon": x, "lat": y} for x, y in
            [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]]


class OSMSourceTests(unittest.TestCase):
    def test_relation_preserves_courtyard_and_suppresses_member_way(self):
        outer, inner = ring(1, 1, 4, 4), ring(2, 2, 3, 3)
        data = {"elements": [
            {"type": "way", "id": 10, "tags": {"building": "yes"}, "geometry": outer},
            {"type": "relation", "id": 20, "tags": {"building": "school"}, "members": [
                {"type": "way", "ref": 10, "role": "outer", "geometry": outer},
                {"type": "way", "ref": 11, "role": "inner", "geometry": inner}]}]}
        buildings, _ = parse_osm(data, BOUNDS)
        self.assertEqual(len(buildings), 1)
        self.assertEqual(shape(buildings[0]["geometry"]).area, 8)
        self.assertEqual(buildings[0]["id"], "osm_relation_20")
        self.assertEqual(buildings[0]["properties"]["height_source"], "display_default")

    def test_road_keeps_routing_attributes_and_connectivity(self):
        data = {"elements": [{"type": "way", "id": 30, "nodes": [1, 2],
            "geometry": [{"lon": 1, "lat": 1}, {"lon": 2, "lat": 2}],
            "tags": {"highway": "residential", "oneway": "-1", "bridge": "yes", "layer": "1"}}]}
        _, roads = parse_osm(data, BOUNDS)
        props = roads[0]["properties"]
        self.assertEqual(props["osm_nodes"], [1, 2])
        self.assertEqual(props["oneway"], "-1")
        self.assertEqual(props["layer"], "1")

    def test_fallback_does_not_duplicate_original_buildings(self):
        def feature(i, coords):
            return {"type": "way", "id": i, "geometry": coords, "tags": {"building": "yes"}}
        originals, _ = parse_osm({"elements": [feature(1, ring(1, 1, 2, 2))]}, BOUNDS)
        fallback, _ = parse_osm({"elements": [feature(2, ring(1, 1, 2, 2)), feature(3, ring(3, 3, 4, 4))]}, BOUNDS)
        result = supplement_buildings(originals, fallback)
        self.assertEqual([f["id"] for f in result], ["osm_way_1", "osm_way_3"])

    def test_partial_overpass_response_is_rejected_and_not_cached(self):
        _CACHE.clear()
        with patch("osm_source.requests.get") as post:
            post.return_value.json.return_value = {"elements": [], "remark": "runtime error: timeout"}
            result = fetch_osm_source(BOUNDS)
        self.assertIsNotNone(result["error"])
        self.assertEqual(_CACHE, {})


if __name__ == "__main__":
    unittest.main()
