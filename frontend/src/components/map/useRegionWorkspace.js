import { useCallback, useRef, useState } from 'react';

export default function useRegionWorkspace({ geoJson, setGeoJson, setRoadsGeoJson, setViewState, setError, setActiveRegionName, cancelAnalysis, backendUrl }) {
  const [region, setRegion] = useState(null);
  const [regions, setRegions] = useState([]);
  const [name, setName] = useState('Çalışma alanı');
  const [dirty, setDirty] = useState(false);
  const [busy, setBusy] = useState(false);
  const [selectedId, setSelectedId] = useState(null);
  const [mode, setMode] = useState('select');
  const [points, setPoints] = useState([]);
  const [vertex, setVertex] = useState(null);
  const [coverageReviewed, setCoverageReviewed] = useState(false);
  const [undoCount, setUndoCount] = useState(0);
  const history = useRef([]);
  const selected = geoJson?.features?.find(f => f.id === selectedId) ?? null;
  const loadRegions = useCallback(async () => {
    try {
      const response = await fetch(`${backendUrl}/api/regions`);
      if (response.ok) setRegions((await response.json()).regions);
    } catch { /* Opening a specific region reports connection errors to the user. */ }
  }, [backendUrl]);

  const showCollection = useCallback(collection => {
    setGeoJson(collection);
    setRoadsGeoJson({ type: 'FeatureCollection', features: collection.features.filter(f => f.properties?.type === 'road') });
  }, [setGeoJson, setRoadsGeoJson]);

  const reset = useCallback(() => {
    setRegion(null); setName('Çalışma alanı'); setDirty(false); setSelectedId(null);
    setMode('select'); setPoints([]); setVertex(null); setCoverageReviewed(false);
    history.current = []; setUndoCount(0);
  }, []);

  const openRegion = useCallback(async (id = 'bartin') => {
    if (dirty) { setError('Önce değişiklikleri kaydedin veya geri alın.'); return; }
    cancelAnalysis(); setBusy(true); setError('');
    try {
      const response = await fetch(`${backendUrl}/api/regions/${id}`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Bölge açılamadı.');
      reset(); setRegion(data); setName(data.name); setActiveRegionName(data.name);
      setCoverageReviewed(data.quality.coverage_reviewed);
      showCollection(data.collection);
      const b = data.collection.meta.bounds;
      setViewState(v => ({ ...v, latitude: (b.north + b.south) / 2, longitude: (b.east + b.west) / 2, zoom: 16.5, pitch: 0, bearing: 0 }));
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }, [dirty, cancelAnalysis, backendUrl, reset, showCollection, setActiveRegionName, setError, setViewState]);

  const change = useCallback(next => {
    if (!geoJson) return;
    cancelAnalysis();
    history.current = [...history.current.slice(-29), { collection: geoJson, coverageReviewed, dirty }];
    setUndoCount(history.current.length);
    setCoverageReviewed(false); setDirty(true); setError('');
    showCollection({ ...next, meta: { ...next.meta, coverage_verified: false, review_status: 'unreviewed' } });
  }, [geoJson, coverageReviewed, dirty, cancelAnalysis, showCollection, setError]);

  const undo = () => {
    const previous = history.current.pop();
    if (!previous) return;
    showCollection(previous.collection); setDirty(previous.dirty); setCoverageReviewed(previous.coverageReviewed);
    setUndoCount(history.current.length); setMode('select'); setPoints([]); setVertex(null);
  };

  const patchSelected = (properties, geometry = null) => {
    if (!selected) return;
    change({ ...geoJson, features: geoJson.features.map(f => f.id === selectedId ? {
      ...f, ...(geometry ? { geometry } : {}),
      properties: { ...f.properties, ...properties,
        ...(geometry ? { geometry_source: 'manual', osm_nodes: undefined, review_status: 'unreviewed', ai_confidence: null } : {}) },
    } : f) });
  };

  const beginDrawing = (kind, replace = false) => {
    if (!geoJson) { setError('Önce bir bölge açın veya tarayın.'); return; }
    cancelAnalysis(); setMode(`${replace ? 'replace' : 'add'}_${kind}`); setPoints([]); setVertex(null);
  };

  const finishDrawing = () => {
    const building = mode.endsWith('building');
    if (points.length < (building ? 3 : 2)) { setError(building ? 'En az üç köşe seçin.' : 'En az iki nokta seçin.'); return; }
    const geometry = { type: building ? 'Polygon' : 'LineString', coordinates: building ? [[...points, points[0]]] : points };
    if (mode.startsWith('replace') && selected) patchSelected({}, geometry);
    else {
      const feature = { type: 'Feature', id: `manual_${crypto.randomUUID()}`, geometry,
        properties: { type: building ? 'building' : 'road', source: 'manual', geometry_source: 'manual',
          name: building ? 'Yeni bina' : 'Yeni yol', review_status: 'unreviewed',
          ...(building ? { height: 9, height_source: 'display_default', floors: null } : { width: 3, highway: 'residential', class: 'residential', oneway: 'no' }) } };
      change({ ...geoJson, features: [...geoJson.features, feature] }); setSelectedId(feature.id);
    }
    setMode('select'); setPoints([]);
  };

  const onMapClick = info => {
    if (mode === 'move_feature' && selected && info.coordinate) {
      const all = [];
      const collect = c => typeof c[0] === 'number' ? all.push(c) : c.forEach(collect);
      collect(selected.geometry.coordinates);
      const xs = all.map(p => p[0]), ys = all.map(p => p[1]);
      const dx = info.coordinate[0] - (Math.min(...xs) + Math.max(...xs)) / 2;
      const dy = info.coordinate[1] - (Math.min(...ys) + Math.max(...ys)) / 2;
      const move = c => typeof c[0] === 'number' ? [c[0] + dx, c[1] + dy] : c.map(move);
      patchSelected({}, { ...selected.geometry, coordinates: move(selected.geometry.coordinates) });
      setMode('select'); return;
    }
    if (mode === 'move_vertex' && vertex && selected && info.coordinate) {
      const geometry = structuredClone(selected.geometry);
      let ring = geometry.coordinates;
      for (const i of vertex.slice(0, -1)) ring = ring[i];
      const index = vertex.at(-1);
      const closed = selected.properties.type === 'building';
      ring[index] = info.coordinate.slice(0, 2);
      if (closed && index === 0) ring[ring.length - 1] = ring[0];
      patchSelected({}, geometry); setVertex(null); setMode('select'); return;
    }
    if (mode.startsWith('add_') || mode.startsWith('replace_')) {
      if (info.coordinate) setPoints(p => [...p, info.coordinate.slice(0, 2)]);
      return;
    }
    if (info.object?.vertexPath) { setVertex(info.object.vertexPath); setMode('move_vertex'); return; }
    setSelectedId(info.object?.id ?? null); setVertex(null); setMode('select');
  };

  const save = async () => {
    if (!geoJson || mode !== 'select') { setError('Önce çizimi tamamlayın veya iptal edin.'); return; }
    cancelAnalysis(); setBusy(true); setError('');
    try {
      const id = region?.id ?? `study_${crypto.randomUUID()}`;
      const response = await fetch(`${backendUrl}/api/regions/${id}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ revision: region?.revision ?? 0, name, collection: geoJson, coverage_reviewed: coverageReviewed }) });
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Kayıt doğrulanamadı.');
      setRegion(data); showCollection(data.collection); setDirty(false); history.current = []; setUndoCount(0);
      setRegions(items => [...items.filter(item => item.id !== data.id), { id: data.id, name: data.name }]);
      setActiveRegionName(data.name);
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  };

  const focusFeature = feature => {
    const values = [];
    const collect = c => typeof c[0] === 'number' ? values.push(c) : c.forEach(collect);
    collect(feature.geometry.coordinates);
    const xs = values.map(p => p[0]), ys = values.map(p => p[1]);
    setViewState(v => ({ ...v, latitude: (Math.min(...ys) + Math.max(...ys)) / 2,
      longitude: (Math.min(...xs) + Math.max(...xs)) / 2, zoom: 19, pitch: 0, bearing: 0 }));
  };
  const nextUnreviewed = () => {
    const all = geoJson?.features ?? [], index = all.findIndex(f => f.id === selectedId);
    const feature = [...all.slice(index + 1), ...all.slice(0, index + 1)].find(f => f.properties?.review_status !== 'reviewed');
    if (feature) { setSelectedId(feature.id); focusFeature(feature); }
    else setError('Denetlenmemiş nesne kalmadı; bölgenin boş alanlarında eksik nesne olup olmadığını da kontrol edin.');
  };

  const removeSelected = () => {
    change({ ...geoJson, features: geoJson.features.filter(f => f.id !== selectedId) });
    setSelectedId(null); setMode('select'); setPoints([]);
  };

  const exportData = () => {
    if (!geoJson) return;
    const blob = new Blob([JSON.stringify({ ...geoJson, meta: { ...geoJson.meta, name, coverage_reviewed: coverageReviewed } }, null, 2)], { type: 'application/geo+json' });
    const url = URL.createObjectURL(blob), a = document.createElement('a');
    a.href = url; a.download = `${region?.id ?? 'calisma-alani'}.geojson`; a.click(); URL.revokeObjectURL(url);
  };

  const restorePrevious = async () => {
    if (!region || region.revision < 2) return;
    setBusy(true); setError('');
    try {
      const response = await fetch(`${backendUrl}/api/regions/${region.id}?revision=${region.revision - 1}`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Önceki sürüm açılamadı.');
      change(data.collection); setSelectedId(null);
    } catch (error) { setError(error.message); }
    finally { setBusy(false); }
  };

  return { region, regions, loadRegions, name, setName: value => { setName(value); setDirty(true); }, dirty, busy, selected, selectedId,
    mode, points, vertex, undoCount, openRegion, reset, onMapClick, beginDrawing, finishDrawing,
    cancelDrawing: () => { setMode('select'); setPoints([]); setVertex(null); },
    removeLastPoint: () => setPoints(p => p.slice(0, -1)),
    patchSelected, removeSelected, undo, save, nextUnreviewed, focusFeature, exportData, restorePrevious,
    beginMove: () => { cancelAnalysis(); setMode('move_feature'); setPoints([]); },
    coverageReviewed, setCoverageReviewed: value => { cancelAnalysis(); setCoverageReviewed(value); setDirty(true); } };
}
