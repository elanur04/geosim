const mercator = lat => Math.asinh(Math.tan(lat * Math.PI / 180));

export function pixelToGeo(point, bounds, width, height) {
  const north = mercator(bounds.north), south = mercator(bounds.south);
  const y = north + (south - north) * point[1] / height;
  return [bounds.west + (bounds.east - bounds.west) * point[0] / width, Math.atan(Math.sinh(y)) * 180 / Math.PI];
}

export function geoToPixel(point, bounds, width, height) {
  return [(point[0] - bounds.west) / (bounds.east - bounds.west) * width,
    (mercator(bounds.north) - mercator(point[1])) / (mercator(bounds.north) - mercator(bounds.south)) * height];
}

export function referenceCollection(buildings, manifest, complete, review = {}) {
  const [width, height] = manifest.image_size;
  return { type: 'FeatureCollection', meta: { role: 'reference', annotation_method: 'manual_from_image',
    analysis_run_id: manifest.id, image_sha256: manifest.artifacts['image.png'].sha256,
    bounds: manifest.bounds, reference_complete: complete,
    dataset_role: review.dataset_role ?? 'unassigned',
    second_reviewer_fraction: Number(review.second_reviewer_fraction ?? 0),
    second_reviewer_area_fraction: Number(review.second_reviewer_area_fraction ?? 0),
    second_reviewer: review.second_reviewer ?? '', reviewed_at: review.reviewed_at || null },
  features: buildings.map(b => ({ type: 'Feature', id: b.id,
    properties: { type: 'building', source: 'manual', review_status: 'reviewed' },
    geometry: { type: 'Polygon', coordinates: b.rings.map(ring => {
      const coords = ring.map(p => pixelToGeo(p, manifest.bounds, width, height));
      return [...coords, coords[0]];
    }) } })) };
}

export function referenceBuildings(collection, manifest) {
  const [width, height] = manifest.image_size;
  return collection.features.map(f => {
    if (f.geometry.type !== 'Polygon') throw new Error('Bu ekran yalnızca Polygon referansları düzenler; çok parçalı nesne için GeoJSON dosyasını kullanın.');
    return { id: f.id, rings: f.geometry.coordinates.map(ring => ring.slice(0, -1).map(p => geoToPixel(p, manifest.bounds, width, height))) };
  });
}
