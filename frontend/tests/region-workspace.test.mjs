import assert from 'node:assert/strict';
import { afterEach, test } from 'node:test';
import React, { act, useCallback, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { JSDOM } from 'jsdom';
import useRegionWorkspace from '../src/components/map/useRegionWorkspace.js';

globalThis.IS_REACT_ACT_ENVIRONMENT = true;
let root, dom, state;
const originalFetch = globalThis.fetch;
const initial = () => ({ type: 'FeatureCollection', meta: { bounds: { west: 32, south: 41, east: 33, north: 42 } }, features: [
  { type: 'Feature', id: 'building', properties: { type: 'building', review_status: 'reviewed' },
    geometry: { type: 'Polygon', coordinates: [[[32.1, 41.1], [32.2, 41.1], [32.2, 41.2], [32.1, 41.1]]] } },
  { type: 'Feature', id: 'road', properties: { type: 'road', review_status: 'unreviewed', osm_nodes: [1, 2] },
    geometry: { type: 'LineString', coordinates: [[32.1, 41.1], [32.2, 41.2]] } },
] });

async function mount() {
  dom = new JSDOM('<div id="root"></div>', { url: 'http://localhost' });
  globalThis.window = dom.window; globalThis.document = dom.window.document;
  function Harness() {
    const [geoJson, setGeoJson] = useState(initial);
    const [roads, setRoadsGeoJson] = useState(null);
    const [view, setViewState] = useState({});
    const [error, setError] = useState('');
    const [, setActiveRegionName] = useState('');
    const cancelAnalysis = useCallback(() => {}, []);
    const editor = useRegionWorkspace({ geoJson, setGeoJson, setRoadsGeoJson, setViewState, setError,
      setActiveRegionName, cancelAnalysis, backendUrl: 'http://localhost:8000' });
    state = { ...editor, collection: geoJson, roads, view, error };
    return null;
  }
  root = createRoot(document.getElementById('root'));
  await act(() => root.render(React.createElement(Harness)));
}

afterEach(async () => {
  if (root) await act(() => root.unmount());
  dom?.window.close(); root = null; dom = null; globalThis.fetch = originalFetch;
});

test('draw building and road, synchronize layers and undo', async () => {
  await mount();
  await act(() => state.beginDrawing('building'));
  for (const coordinate of [[32.3, 41.3], [32.4, 41.3], [32.4, 41.4]]) await act(() => state.onMapClick({ coordinate }));
  await act(() => state.finishDrawing());
  const added = state.collection.features.at(-1);
  assert.equal(added.properties.type, 'building');
  assert.deepEqual(added.geometry.coordinates[0][0], added.geometry.coordinates[0].at(-1));
  assert.equal(state.dirty, true);
  await act(() => state.undo());
  assert.equal(state.collection.features.length, 2);
  assert.equal(state.dirty, false);
  await act(() => state.beginDrawing('road'));
  for (const coordinate of [[32.3, 41.3], [32.4, 41.4]]) await act(() => state.onMapClick({ coordinate }));
  await act(() => state.finishDrawing());
  assert.equal(state.roads.features.length, 2);
});

test('moving a polygon vertex preserves ring closure and resets review', async () => {
  await mount();
  await act(() => state.onMapClick({ object: { id: 'building' } }));
  await act(() => state.onMapClick({ object: { vertexPath: [0, 0] } }));
  await act(() => state.onMapClick({ coordinate: [32.11, 41.11] }));
  const f = state.selected, ring = f.geometry.coordinates[0];
  assert.deepEqual(ring[0], [32.11, 41.11]);
  assert.deepEqual(ring.at(-1), ring[0]);
  assert.equal(f.properties.review_status, 'unreviewed');
  assert.equal(f.properties.geometry_source, 'manual');
});

test('whole object movement retains shape and removes obsolete OSM nodes', async () => {
  await mount();
  await act(() => state.onMapClick({ object: { id: 'road' } }));
  await act(() => state.beginMove());
  await act(() => state.onMapClick({ coordinate: [32.5, 41.5] }));
  const pts = state.selected.geometry.coordinates;
  assert.ok(Math.abs(pts[1][0] - pts[0][0] - 0.1) < 1e-10);
  assert.equal((pts[0][0] + pts[1][0]) / 2, 32.5);
  assert.equal(state.selected.properties.osm_nodes, undefined);
});

test('delete updates road layer and undo restores it', async () => {
  await mount();
  await act(() => state.onMapClick({ object: { id: 'road' } }));
  await act(() => state.removeSelected());
  assert.equal(state.collection.features.length, 1);
  assert.equal(state.roads.features.length, 0);
  await act(() => state.undo());
  assert.equal(state.roads.features.length, 1);
});

test('save conflict retains unsaved corrections', async () => {
  await mount();
  await act(() => state.onMapClick({ object: { id: 'building' } }));
  await act(() => state.patchSelected({ name: 'Corrected building' }));
  globalThis.fetch = async () => ({ ok: false, json: async () => ({ detail: 'Version conflict' }) });
  await act(() => state.save());
  assert.equal(state.dirty, true);
  assert.equal(state.error, 'Version conflict');
  assert.equal(state.selected.properties.name, 'Corrected building');
});

test('saved region opens without analyze request and mutations clear coverage approval', async () => {
  await mount();
  const calls = [];
  globalThis.fetch = async url => { calls.push(url); return { ok: true, json: async () => ({ id: 'bartin', name: 'Bartin', revision: 1,
    collection: initial(), quality: { coverage_reviewed: true } }) }; };
  await act(() => state.openRegion('bartin'));
  assert.deepEqual(calls, ['http://localhost:8000/api/regions/bartin']);
  assert.equal(state.coverageReviewed, true);
  await act(() => state.onMapClick({ object: { id: 'road' } }));
  await act(() => state.patchSelected({ width: 4 }));
  assert.equal(state.coverageReviewed, false);
  await act(() => state.openRegion('bartin'));
  assert.equal(calls.length, 1, 'dirty work must not be replaced');
});
