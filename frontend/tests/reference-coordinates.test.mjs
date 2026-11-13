import assert from 'node:assert/strict';
import { test } from 'node:test';
import { geoToPixel, pixelToGeo, referenceBuildings, referenceCollection } from '../src/components/map/referenceCoordinates.js';

const manifest = { id: 'run', image_size: [2560, 2560], bounds: { west: 32.33, east: 32.34, south: 41.63, north: 41.64 },
  artifacts: { 'image.png': { sha256: 'fixed-image-hash' } } };

test('pixel and geographic coordinates round trip across the full image', () => {
  for (const p of [[0, 0], [2560, 2560], [17.5, 1900.2], [1280, 1280]]) {
    const actual = geoToPixel(pixelToGeo(p, manifest.bounds, 2560, 2560), manifest.bounds, 2560, 2560);
    assert.ok(Math.abs(actual[0] - p[0]) < 1e-6);
    assert.ok(Math.abs(actual[1] - p[1]) < 1e-6);
  }
});

test('manual reference retains courtyard rings and exact image identity', () => {
  const buildings = [{ id: 'manual1', rings: [
    [[100, 100], [900, 100], [900, 900], [100, 900]],
    [[300, 300], [500, 300], [500, 500], [300, 500]],
  ] }];
  const data = referenceCollection(buildings, manifest, true);
  assert.equal(data.meta.role, 'reference');
  assert.equal(data.meta.image_sha256, 'fixed-image-hash');
  assert.equal(data.features[0].properties.source, 'manual');
  assert.equal(data.features[0].geometry.coordinates.length, 2);
  const restored = referenceBuildings(data, manifest);
  for (let r = 0; r < 2; r++) for (let p = 0; p < 4; p++) for (let k = 0; k < 2; k++)
    assert.ok(Math.abs(restored[0].rings[r][p][k] - buildings[0].rings[r][p][k]) < 1e-6);
});

test('incomplete manual annotation cannot accidentally claim full reference coverage', () => {
  const data = referenceCollection([], manifest, false);
  assert.equal(data.meta.reference_complete, false);
  assert.deepEqual(data.features, []);
});
