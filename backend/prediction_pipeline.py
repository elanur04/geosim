"""Tüm model sağlayıcılarının ortak kullandığı, geometriyi koruyan son işleme adımları."""
import copy
from shapely.geometry import Polygon

PIPELINE_VERSION = 'geometry-preserving-v1'
PROFILES = ('conservative', 'raw', 'legacy')


def deduplicate_predictions(predictions, iou_threshold=0.5):
    """Yinelenen maskeleri ayıklar; bir bina yalnızca büyük olduğu için elenmez.

    Kutu tabanlı NMS kullanılmaz: bitişik ve içbükey binaların kutuları örtüşebilir.
    Karo bilgisi varsa yalnızca farklı karolardan gelen tahminler birbirini bastırabilir.
    Kenarda kesilmiş parçalar yerine tam maskeler tercih edilir.
    """
    kept, geometries = [], []
    order = sorted(predictions, key=lambda p: (bool(p.get('_cut')), -float(p.get('confidence', 0))))
    for pred in order:
        points = pred.get('points', [])
        try:
            geom = Polygon([(p['x'], p['y']) if isinstance(p, dict) else p for p in points])
            if geom.is_empty or not geom.is_valid or geom.area <= 0:
                continue
            duplicate = False
            for previous, other in zip(kept, geometries):
                if pred.get('tile') is not None and pred.get('tile') == previous.get('tile'):
                    continue
                intersection = geom.intersection(other).area
                union = geom.area + other.area - intersection
                # Komşu karodaki tam maskenin içinde kalan kesik parçalar.
                contained_cut = pred.get('_cut') and not previous.get('_cut') and intersection / geom.area >= .9
                if intersection / union >= iou_threshold or contained_cut:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(copy.deepcopy(pred))
                geometries.append(geom)
        except (ValueError, TypeError, KeyError):
            continue
    return kept


def preserve_polygon(points, tolerance=0):
    """Özgün topolojiyi korur; sadeleştirme yalnızca maske IoU >= .98 ise kabul edilir."""
    if len(points) < 3:
        return []
    polygon = Polygon(points)
    if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
        return []
    simplified = polygon.simplify(max(0, tolerance), preserve_topology=True)
    if (simplified.geom_type == 'Polygon' and simplified.is_valid
            and polygon.intersection(simplified).area / polygon.union(simplified).area >= .98):
        polygon = simplified
    return [[float(x), float(y)] for x, y in polygon.exterior.coords[:-1]]
