/**
 * DeckMapView.jsx — Mapbox + Deck.gl 3D Kentsel Simülasyon Haritası
 */

import React, { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import Map from 'react-map-gl/mapbox';
import DeckGL from '@deck.gl/react';
import { GeoJsonLayer, PathLayer, ScatterplotLayer } from '@deck.gl/layers';
import { FlyToInterpolator, COORDINATE_SYSTEM } from '@deck.gl/core';
import useTrafficSimulation from '../../simulation/useTrafficSimulation';
import SimulationPanel from './SimulationPanel';
import 'mapbox-gl/dist/mapbox-gl.css';
import useRegionWorkspace from './useRegionWorkspace';
import RegionEditor from './RegionEditor';

const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN || 'TODO_MAPBOX_TOKEN_BURAYA';
const BACKEND_URL  = 'http://localhost:8000';

// Bina tespitinde kullanılan model ve güven eşiği
const MODEL_PROVIDER   = 'roboflow';
const MODEL_CONFIDENCE = 0.4;

// Bina renkleri kaynağa göre: OSM, yapay zekâ ile tespit edilen, elle çizilen
const SOURCE_STYLES = {
  osm:    { label: 'OSM binası',             rgb: [56, 189, 248] },
  ai:     { label: 'Yapay zekâ ile eklenen', rgb: [74, 222, 128] },
  manual: { label: 'Elle çizilen',           rgb: [250, 204, 21] },
};
const ROAD_RGB = [255, 140, 0];

function buildingSource(properties) {
  const source = properties?.source;
  if (source === 'osm') return 'osm';
  if (source === 'manual') return 'manual';
  return 'ai';
}

const toCss = ([r, g, b]) => `rgb(${r}, ${g}, ${b})`;

// Tarayıcının ağ hatalarını kullanıcıya anlaşılır bir mesajla göster
function friendlyError(message) {
  if (!message) return '';
  if (/failed to fetch|networkerror|load failed/i.test(message)) {
    return 'Sunucuya bağlanılamadı. Backend\'in çalıştığından emin olun (http://localhost:8000).';
  }
  return message;
}

// Mapbox uydu karoları reddedildiğinde (ör. 429 Too Many Requests) kullanılacak yedek altlık
const ESRI_SATELLITE_STYLE = {
  version: 8,
  sources: {
    esri: {
      type: 'raster',
      tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'],
      tileSize: 256,
      maxzoom: 19,
      attribution: 'Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics',
    },
  },
  layers: [{ id: 'esri-satellite', type: 'raster', source: 'esri' }],
};

const INITIAL_VIEW = {
  longitude:  32.336,
  latitude:   41.634,
  zoom:       16.5,
  pitch:      0,
  bearing:    0,
};

// Ekranda görüntülenen bölgenin Bounding Box (south, west, north, east) sınırlarını hesaplar
function getViewportBoundingBox(centerLat, centerLon, zoom, imgW = 1280, imgH = 1280) {
  const tileSize = 512;
  const worldSize = tileSize * Math.pow(2, zoom);
  const cx = ((centerLon + 180.0) / 360.0) * worldSize;
  const sinLat = Math.min(Math.max(Math.sin((centerLat * Math.PI) / 180.0), -0.9999), 0.9999);
  const cy = (0.5 - Math.log((1.0 + sinLat) / (1.0 - sinLat)) / (4.0 * Math.PI)) * worldSize;

  const halfW = imgW / 2.0;
  const halfH = imgH / 2.0;

  const west = ((cx - halfW) / worldSize) * 360.0 - 180.0;
  const east = ((cx + halfW) / worldSize) * 360.0 - 180.0;

  const nNorth = Math.PI - (2.0 * Math.PI * (cy - halfH)) / worldSize;
  const north = (Math.atan(Math.sinh(nNorth)) * 180.0) / Math.PI;

  const nSouth = Math.PI - (2.0 * Math.PI * (cy + halfH)) / worldSize;
  const south = (Math.atan(Math.sinh(nSouth)) * 180.0) / Math.PI;

  return {
    south: Number(south.toFixed(7)),
    west:  Number(west.toFixed(7)),
    north: Number(north.toFixed(7)),
    east:  Number(east.toFixed(7)),
  };
}

// ── Yardımcı Bileşenler ────────────────────────────────────────────────────────
function LiveClock() {
  const [time, setTime] = useState(new Date());
  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const dateStr = time.toLocaleDateString('tr-TR', { day: '2-digit', month: '2-digit', year: 'numeric' });
  const timeStr = time.toLocaleTimeString('tr-TR', { hour12: false });
  return <span>{dateStr} | {timeStr}</span>;
}

function MetricBox({ label, val, color }) {
  return (
    <div style={{ background: 'rgba(0,0,0,0.5)', border: `1px solid ${color}`, padding: '8px 6px', textAlign: 'center', borderRadius: '4px' }}>
      <div className="mono-font" style={{ fontSize: '18px', fontWeight: 700, color: color, lineHeight: 1 }}>
        {val}
      </div>
      <div style={{ fontSize: '10px', color: '#8ba4b8', marginTop: '6px', letterSpacing: '1px' }}>{label}</div>
    </div>
  );
}

const PRESET_LOCATIONS = [
  { name: 'FATİH / BALAT', desc: 'İstanbul', lat: 41.0295, lon: 28.9485, zoom: 17.0 },
  { name: 'KADIKÖY / MODA', desc: 'İstanbul', lat: 40.9855, lon: 29.0275, zoom: 17.0 },
  { name: 'SULTANAHMET', desc: 'İstanbul', lat: 41.0055, lon: 28.9765, zoom: 17.0 },
];

function Legend() {
  const items = [
    ...Object.values(SOURCE_STYLES).map(s => ({ label: s.label, color: toCss(s.rgb), box: true })),
    { label: 'Yol ağı', color: toCss(ROAD_RGB), box: false },
  ];
  return (
    <div className="hud-panel" style={{ position: 'absolute', top: '64px', left: '370px', padding: '8px 12px', zIndex: 10, fontSize: '12px', display: 'flex', gap: '14px', flexWrap: 'wrap' }}>
      {items.map(item => (
        <span key={item.label} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={item.box
            ? { width: 12, height: 12, background: item.color.replace('rgb', 'rgba').replace(')', ', 0.35)'), border: `2px solid ${item.color}` }
            : { width: 16, height: 3, background: item.color }} />
          {item.label}
        </span>
      ))}
    </div>
  );
}

// ── Ana Bileşen ───────────────────────────────────────────────────────────────
export default function DeckMapView() {
  const [viewState,          setViewState]          = useState(INITIAL_VIEW);
  const [basemap,            setBasemap]            = useState('mapbox');
  const [geoJson,            setGeoJson]            = useState(null);
  const [roadsGeoJson,       setRoadsGeoJson]       = useState(null);
  const [isAnalyzing,        setIsAnalyzing]        = useState(false);
  const [isAiEnriching,      setIsAiEnriching]      = useState(false);
  const [isLoadingRoads,     setIsLoadingRoads]     = useState(false);
  const [error,              setErrorRaw]           = useState('');
  const setError = useCallback(message => setErrorRaw(friendlyError(message)), []);
  const [tooltip,           setTooltip]            = useState(null);
  const [inputLat,           setInputLat]           = useState(INITIAL_VIEW.latitude.toFixed(6));
  const [inputLon,           setInputLon]           = useState(INITIAL_VIEW.longitude.toFixed(6));

  // Arama (Geocoding) state'leri
  const [searchQuery,        setSearchQuery]        = useState('');
  const [searchResults,      setSearchResults]      = useState([]);
  const [isSearching,        setIsSearching]        = useState(false);
  const [showSearchDropdown, setShowSearchDropdown] = useState(false);
  const [activeRegionName,   setActiveRegionName]   = useState('');

  const aiAbortRef         = useRef(null);
  const lastHoverTextRef   = useRef(null);
  const isEditingCoordsRef = useRef(false);
  const searchBoxRef       = useRef(null);
  const cancelAnalysis = useCallback(() => {
    aiAbortRef.current?.abort();
    setIsAnalyzing(false); setIsAiEnriching(false);
  }, []);
  const editor = useRegionWorkspace({ geoJson, setGeoJson, setRoadsGeoJson, setViewState, setError,
    setActiveRegionName, cancelAnalysis, backendUrl: BACKEND_URL });
  const { region: savedRegion, dirty: hasEdits, reset: resetWorkspace } = editor;

  // Kullanıcı koordinat kutularında yazmıyorsa harita merkezini kutulara yansıt
  useEffect(() => {
    if (!isEditingCoordsRef.current) {
      setInputLat(viewState.latitude.toFixed(6));
      setInputLon(viewState.longitude.toFixed(6));
    }
  }, [viewState.latitude, viewState.longitude]);

  // Arama kutusu dışına tıklandığında açılır listeyi kapat
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (searchBoxRef.current && !searchBoxRef.current.contains(e.target)) {
        setShowSearchDropdown(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const updateHoverTooltip = useCallback((nextText) => {
    if (lastHoverTextRef.current === nextText) return;
    lastHoverTextRef.current = nextText;
    setTooltip(nextText ? { text: nextText } : null);
  }, []);

  // ── Yol Ağlarını Çek (/api/roads) ───────────────────────────────────────────
  const fetchRoads = useCallback(async (customCoords = null) => {
    if (savedRegion || hasEdits) {
      setError('Kayıtlı çalışma alanındaki yollar korunuyor. Eksik yolu sağ panelden çizip kaydedebilirsiniz.');
      return;
    }
    setIsLoadingRoads(true);
    const parsedLat  = parseFloat(String(inputLat).replace(',', '.'));
    const parsedLon  = parseFloat(String(inputLon).replace(',', '.'));
    const targetLat  = customCoords?.lat  ?? (Number.isFinite(parsedLat) ? parsedLat : viewState.latitude);
    const targetLon  = customCoords?.lon  ?? (Number.isFinite(parsedLon) ? parsedLon : viewState.longitude);
    const targetZoom = customCoords?.zoom ?? (viewState.zoom < 16.0 ? 17.0 : viewState.zoom);

    const bbox = getViewportBoundingBox(targetLat, targetLon, targetZoom);
    const params = new URLSearchParams({
      south: bbox.south,
      west:  bbox.west,
      north: bbox.north,
      east:  bbox.east,
      lat:   targetLat,
      lon:   targetLon,
      zoom:  targetZoom,
    });

    try {
      const response = await fetch(`${BACKEND_URL}/api/roads?${params.toString()}`);
      if (!response.ok) {
        throw new Error(`Yol verisi alınamadı: ${response.status}`);
      }
      const data = await response.json();
      setRoadsGeoJson(data);
      if (data.meta?.overpass_error) setError('Yol verisi eksik: ' + data.meta.overpass_error);
    } catch (err) {
      console.error('Yol ağı çekme hatası:', err);
      setError(err.message || 'Yol verisi alınamadı.');
    } finally {
      setIsLoadingRoads(false);
    }
  }, [viewState.latitude, viewState.longitude, viewState.zoom, inputLat, inputLon, savedRegion, hasEdits, setError]);

  // ── 2 Aşamalı Hızlı Analiz (1. Aşama: Anında OSM/MVT Poligonları, 2. Aşama: Arka Planda Roboflow AI) ──
  const handleAnalyze = useCallback(async (customCoords = null) => {
    if (hasEdits) { setError('Yeni taramadan önce düzenlemeleri kaydedin veya geri alın.'); return; }
    resetWorkspace();
    if (aiAbortRef.current) {
      aiAbortRef.current.abort();
    }
    const controller = new AbortController();
    aiAbortRef.current = controller;

    setIsAnalyzing(true);
    setError('');

    const parsedLat  = parseFloat(String(inputLat).replace(',', '.'));
    const parsedLon  = parseFloat(String(inputLon).replace(',', '.'));
    const targetLat  = customCoords?.lat  ?? (Number.isFinite(parsedLat) ? parsedLat : viewState.latitude);
    const targetLon  = customCoords?.lon  ?? (Number.isFinite(parsedLon) ? parsedLon : viewState.longitude);
    const targetZoom = customCoords?.zoom ?? (viewState.zoom < 16.0 ? 17.0 : viewState.zoom);

    setInputLat(targetLat.toFixed(6));
    setInputLon(targetLon.toFixed(6));

    if (customCoords) {
      // Yeni bir koordinata veya aranan konuma gidilirken önceki bölgenin poligonlarını temizle
      setGeoJson(null);
      setRoadsGeoJson(null);
    }

    if (viewState.zoom < 16.0 || customCoords) {
      setViewState(prev => ({
        ...prev,
        latitude: targetLat,
        longitude: targetLon,
        zoom: targetZoom,
        transitionDuration: 900,
        transitionInterpolator: new FlyToInterpolator(),
      }));
    }

    const tReq = performance.now();
    try {
      // 1. Aşama: Hızlı Vektör Katmanı (include_ai: false) -> ~50-300ms içinde bina ve yolları ekrana çizer
      const fastResp = await fetch(`${BACKEND_URL}/analyze`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({
          latitude:   targetLat,
          longitude:  targetLon,
          zoom:       targetZoom,
          include_ai: false,
        }),
        signal: controller.signal,
      });

      if (!fastResp.ok) {
        const errData = await fastResp.json().catch(() => ({}));
        throw new Error(errData.detail || `Sunucu hatası: ${fastResp.status}`);
      }

      const fastData = await fastResp.json();
      setGeoJson(fastData);
      const fastRoads = fastData.features?.filter(f => f.properties?.type === 'road') ?? [];
      setRoadsGeoJson({ type: 'FeatureCollection', features: fastRoads });
      setIsAnalyzing(false);

      requestAnimationFrame(() => requestAnimationFrame(() => {
        console.table({
          '1. Aşama (Anında Poligon Çizimi) ms': Math.round(performance.now() - tReq),
          'özellik sayısı':                      fastData.features?.length ?? 0,
          'sunucu toplam s':                     fastData.meta?.timings?.total_s,
        });
      }));

      // 2. Aşama: Arka Planda Roboflow AI Zenginleştirmesi (kullanıcıyı bekletmeden)
      setIsAiEnriching(true);
      const aiResp = await fetch(`${BACKEND_URL}/analyze`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({
          latitude:   targetLat,
          longitude:  targetLon,
          zoom:       targetZoom,
          include_ai: true,
          model_provider: MODEL_PROVIDER,
          confidence: MODEL_CONFIDENCE,
          processing_profile: "conservative",
        }),
        signal: controller.signal,
      });

      if (aiResp.ok) {
        const aiData = await aiResp.json();
        setGeoJson(aiData);
        const issues = [];
        if (aiData.meta?.roboflow_error) issues.push(aiData.meta.roboflow_error);
        if (aiData.meta?.osm_source_error) issues.push('OSM sunucusuna ulaşılamadı; Mapbox bina verisi gösteriliyor.');
        if (aiData.meta?.failed_tiles) issues.push(`${aiData.meta.failed_tiles} görüntü parçası taranamadı; sonuç eksik olabilir.`);
        setError(issues.join(' '));
        const aiRoads = aiData.features?.filter(f => f.properties?.type === 'road') ?? [];
        if (aiRoads.length > 0) {
          setRoadsGeoJson({ type: 'FeatureCollection', features: aiRoads });
        }
      } else {
        const detail = await aiResp.json().catch(() => ({}));
        throw new Error(detail.detail || `Ayrıntılı analiz tamamlanamadı: ${aiResp.status}`);
      }
    } catch (err) {
      if (err.name === 'AbortError') return;
      console.error('Analiz hatası:', err);
      setError(err.message || 'Bağlantı hatası.');
    } finally {
      if (aiAbortRef.current === controller) {
        setIsAnalyzing(false);
        setIsAiEnriching(false);
      }
    }
  }, [viewState.latitude, viewState.longitude, viewState.zoom, inputLat, inputLon, hasEdits, resetWorkspace, setError]);

  // İlk açılışta mevcut bölgenin bina ve yol poligonlarını anında yükle
  useEffect(() => {
    editor.openRegion('bartin');
    return () => {
      if (aiAbortRef.current) aiAbortRef.current.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Koordinata Git ve Otomatik Bina & Yol Analizi Başlat ───────────────────
  const handleJump = useCallback(() => {
    isEditingCoordsRef.current = false;
    const lat = parseFloat(String(inputLat).trim().replace(',', '.'));
    const lon = parseFloat(String(inputLon).trim().replace(',', '.'));

    if (!Number.isFinite(lat) || !Number.isFinite(lon) || lat < -90 || lat > 90 || lon < -180 || lon > 180) {
      setError('Geçerli bir enlem (-90..90) ve boylam (-180..180) girin.');
      return;
    }

    setActiveRegionName(`KOORDİNAT (${lat.toFixed(4)}, ${lon.toFixed(4)})`);
    handleAnalyze({ lat, lon, zoom: 17.0 });
  }, [inputLat, inputLon, handleAnalyze, setError]);

  // Koordinat kutusuna "41.0082, 28.9784" gibi tek seferde yapıştırma desteği
  const handleCoordInputChange = (field, rawVal) => {
    if (rawVal.includes(',') && rawVal.split(',').length === 2) {
      const [p1, p2] = rawVal.split(',').map(s => s.trim());
      if (!Number.isNaN(Number(p1)) && !Number.isNaN(Number(p2)) && p1 !== '' && p2 !== '') {
        setInputLat(p1);
        setInputLon(p2);
        return;
      }
    }
    if (field === 'lat') setInputLat(rawVal);
    else setInputLon(rawVal);
  };

  // ── Bölge / Yer Arama Fonksiyonu (Backend + Mapbox Geocoding Fallback) ──────
  const fetchSearchLocations = useCallback(async (queryText) => {
    const q = (queryText || '').trim();
    if (q.length < 2) {
      setSearchResults([]);
      return [];
    }

    setIsSearching(true);
    try {
      // 1. Önce doğrudan koordinat girilmiş mi kontrol et ("41.0082, 28.9784")
      const coordParts = q.replace(';', ',').split(',').map(s => s.trim()).filter(Boolean);
      if (coordParts.length === 2) {
        const cLat = parseFloat(coordParts[0]);
        const cLon = parseFloat(coordParts[1]);
        if (Number.isFinite(cLat) && Number.isFinite(cLon) && cLat >= -90 && cLat <= 90 && cLon >= -180 && cLon <= 180) {
          const directItem = [{
            id:        `coord_${cLat}_${cLon}`,
            name:      `${cLat.toFixed(5)}, ${cLon.toFixed(5)}`,
            full_name: `Koordinat Noktası (${cLat.toFixed(6)}, ${cLon.toFixed(6)})`,
            lat:       cLat,
            lon:       cLon,
            zoom:      17.0,
          }];
          setSearchResults(directItem);
          setShowSearchDropdown(true);
          return directItem;
        }
      }

      // 2. Backend /api/search uç noktasını sorgula
      try {
        const resp = await fetch(`${BACKEND_URL}/api/search?q=${encodeURIComponent(q)}&limit=6`);
        if (resp.ok) {
          const data = await resp.json();
          if (Array.isArray(data.results) && data.results.length > 0) {
            setSearchResults(data.results);
            setShowSearchDropdown(true);
            return data.results;
          }
        }
      } catch {
        // Backend erişilemezse doğrudan Mapbox Geocoding yedeğine geç
      }

      // 3. İstemci tarafı doğrudan Mapbox Geocoding yedeği
      if (MAPBOX_TOKEN && !MAPBOX_TOKEN.startsWith('TODO')) {
        const mbUrl = `https://api.mapbox.com/geocoding/v5/mapbox.places/${encodeURIComponent(q)}.json?access_token=${MAPBOX_TOKEN}&language=tr&limit=6&types=poi,address,neighborhood,locality,place,district,region`;
        const mbResp = await fetch(mbUrl);
        if (mbResp.ok) {
          const mbData = await mbResp.json();
          const mapped = (mbData.features || []).map((feat, idx) => ({
            id:        feat.id || `mb_${idx}`,
            name:      feat.text_tr || feat.text || q,
            full_name: feat.place_name_tr || feat.place_name || q,
            lat:       Number(feat.center[1].toFixed(6)),
            lon:       Number(feat.center[0].toFixed(6)),
            zoom:      17.0,
          }));
          setSearchResults(mapped);
          setShowSearchDropdown(true);
          return mapped;
        }
      }

      setSearchResults([]);
      return [];
    } catch (err) {
      console.error('Konum arama hatası:', err);
      return [];
    } finally {
      setIsSearching(false);
    }
  }, []);

  // Kullanıcı yazarken otomatik öneri listesini getir (220ms debounce)
  useEffect(() => {
    const q = searchQuery.trim();
    if (q.length < 2) {
      setSearchResults([]);
      return;
    }
    const timer = setTimeout(() => {
      fetchSearchLocations(q);
    }, 220);
    return () => clearTimeout(timer);
  }, [searchQuery, fetchSearchLocations]);

  // Arama listesinden bir yer seçildiğinde otomatik olarak oraya git ve bina/yol poligonlarını çıkar
  const handleSelectSearchResult = useCallback((item) => {
    if (!item) return;
    setSearchQuery(item.name);
    setActiveRegionName(item.name.toUpperCase());
    setShowSearchDropdown(false);
    setInputLat(item.lat.toFixed(6));
    setInputLon(item.lon.toFixed(6));
    handleAnalyze({
      lat:  item.lat,
      lon:  item.lon,
      zoom: item.zoom || 17.0,
    });
  }, [handleAnalyze]);

  // "ARA" butonuna tıklandığında veya Enter'a basıldığında
  const handleSearchSubmit = useCallback(async (e) => {
    if (e) e.preventDefault();
    const q = searchQuery.trim();
    if (!q) return;

    let currentResults = searchResults;
    if (!currentResults.length) {
      currentResults = await fetchSearchLocations(q);
    }
    if (currentResults.length > 0) {
      setShowSearchDropdown(true);
      // Eğer tek sonuç varsa veya kullanıcı doğrudan Enter'a bastıysa ilk sonuca git ve analiz et
      if (currentResults.length === 1 || e?.type === 'submit') {
        handleSelectSearchResult(currentResults[0]);
      }
    } else {
      setError(`"${q}" için konum bulunamadı.`);
    }
  }, [searchQuery, searchResults, fetchSearchLocations, handleSelectSearchResult, setError]);

  // ── Vektör Verilerini Memoize Et (her kamera hareketinde yeniden hesaplanmasını önler) ──
  const { buildingCollection, treeCollection, carCollection, roadFeatures, counts } = useMemo(() => {
    const buildings = [];
    const trees     = [];
    const cars      = [];
    const analyzeRoads = [];

    const bySource = { osm: 0, ai: 0, manual: 0 };
    if (geoJson?.features) {
      let bIdx = 0;
      for (const f of geoJson.features) {
        const t = f.properties?.type;
        if (t === 'building') {
          const source = buildingSource(f.properties);
          const rgb = SOURCE_STYLES[source].rgb;
          bySource[source]++;
          buildings.push({
            ...f,
            properties: {
              ...f.properties,
              _idx: bIdx,
              _source: source,
              _fillColor: [rgb[0], rgb[1], rgb[2], 70],
              _lineColor: [rgb[0], rgb[1], rgb[2], 255],
            },
          });
          bIdx++;
        } else if (t === 'road') {
          analyzeRoads.push(f);
        } else if (t === 'tree') {
          trees.push(f);
        } else if (t === 'car') {
          cars.push(f);
        }
      }
    }

    const rawRoads = (roadsGeoJson?.features?.length > 0) ? roadsGeoJson.features : analyzeRoads;
    const roads = rawRoads.flatMap(f => f.geometry.type === 'MultiLineString'
      ? f.geometry.coordinates.map(coordinates => ({ ...f, geometry: { type: 'LineString', coordinates } })) : [f]);

    return {
      buildingCollection: { type: 'FeatureCollection', features: buildings },
      treeCollection:     { type: 'FeatureCollection', features: trees },
      carCollection:      { type: 'FeatureCollection', features: cars },
      roadFeatures:       roads,
      counts: {
        building: buildings.length,
        road:     roads.length,
        osm:      bySource.osm,
        ai:       bySource.ai,
      },
    };
  }, [geoJson, roadsGeoJson]);

  // Trafik simülasyonu: fizik Web Worker'da çalışır; burada yalnızca veri verilir ve kareler çizilir.
  const trafficSim = useTrafficSimulation({ roads: roadFeatures, buildings: buildingCollection.features, enabled: roadFeatures.length > 1 });
  const simFrame = trafficSim.frame;

  const editVertices = useMemo(() => {
    if (!editor.selected || editor.mode !== 'select') return [];
    const found = [];
    const visit = (coords, path = []) => {
      if (typeof coords[0]?.[0] === 'number') {
        const count = coords.length - (editor.selected.properties.type === 'building' ? 1 : 0);
        for (let i = 0; i < count; i++) found.push({ position: coords[i], vertexPath: [...path, i] });
      } else coords.forEach((part, i) => visit(part, [...path, i]));
    };
    visit(editor.selected.geometry.coordinates);
    return found;
  }, [editor.selected, editor.mode]);
  const draftCollection = useMemo(() => {
    const features = editor.points.map(p => ({ type: 'Feature', properties: {}, geometry: { type: 'Point', coordinates: p } }));
    if (editor.points.length > 1) features.push({ type: 'Feature', properties: {}, geometry: { type: 'LineString', coordinates: editor.points } });
    return { type: 'FeatureCollection', features };
  }, [editor.points]);

  // ── Deck.gl Katmanları (Memoize edilmiş) ────────────────
  const layers = useMemo(() => [
    ...(roadFeatures.length > 0 ? [
      new PathLayer({
        id:             'osm-roads-glow-layer',
        data:           roadFeatures,
        getPath:        d => d.geometry.coordinates,
        getColor:       [...ROAD_RGB, 75],
        getWidth:       d => (d.properties?.width ?? 3) + 5,
        widthUnits:     'pixels',
        widthMinPixels: 5,
        widthMaxPixels: 18,
        capRounded:     true,
        jointRounded:   true,
        pickable:       false,
      }),
      new PathLayer({
        id:             'osm-roads-layer',
        data:           roadFeatures,
        getPath:        d => d.geometry.coordinates,
        getColor:       ROAD_RGB,
        getWidth:       3,
        widthUnits:     'pixels',
        widthMinPixels: 2,
        widthMaxPixels: 10,
        capRounded:     true,
        jointRounded:   true,
        pickable:       true,
        autoHighlight:  true,
        highlightColor: [0, 255, 255, 230],
        onHover:        info => updateHoverTooltip(info.object
          ? `Yol: ${info.object.properties?.name || 'isimsiz'} (${info.object.properties?.highway || info.object.properties?.class || 'yol'})`
          : null),
      }),
    ] : []),

    ...(buildingCollection.features.length > 0 ? [
      new GeoJsonLayer({
        id:                 'buildings-layer',
        data:               buildingCollection,
        stroked:            true,
        filled:             true,
        extruded:           false,
        getFillColor:       f => f.properties._fillColor,
        getLineColor:       f => f.properties._lineColor,
        lineWidthMinPixels: 2,
        pickable:           true,
        autoHighlight:      true,
        highlightColor:     [255, 255, 255, 110],
        onHover:            info => updateHoverTooltip(info.object
          ? `Bina #${info.object.properties._idx + 1} · ${SOURCE_STYLES[info.object.properties._source].label}${info.object.properties._source === 'ai' && info.object.properties.ai_confidence != null ? ` (güven %${Math.round(info.object.properties.ai_confidence * 100)})` : ''}`
          : null),
      }),
    ] : []),

    ...(treeCollection.features.length > 0 ? [
      new GeoJsonLayer({
        id:                 'trees-layer',
        data:               treeCollection,
        stroked:            true,
        filled:             true,
        pointType:          'circle',
        getPointRadius:     4,
        pointRadiusUnits:   'pixels',
        getFillColor:       [0, 255, 157, 60],
        getLineColor:       [0, 255, 157, 255],
        lineWidthMinPixels: 1.5,
        pickable:           true,
        onHover:            info => updateHoverTooltip(info.object ? 'Bitki örtüsü' : null),
      }),
    ] : []),

    ...(carCollection.features.length > 0 ? [
      new GeoJsonLayer({
        id:                 'cars-layer',
        data:               carCollection,
        stroked:            true,
        filled:             true,
        pointType:          'circle',
        getPointRadius:     3.5,
        pointRadiusUnits:   'pixels',
        getFillColor:       [255, 0, 85, 120],
        getLineColor:       [255, 0, 85, 255],
        lineWidthMinPixels: 1.5,
        pickable:           true,
        onHover:            info => updateHoverTooltip(info.object ? 'Araç' : null),
      }),
    ] : []),
    // Simülasyon araçları: Worker'dan gelen Float32Array doğrudan GPU özniteliği olarak kullanılır
    // (x, y, yön, hız; graf orijinine göre metre ofseti).
    ...(simFrame && simFrame.count > 0 && ['playing', 'paused'].includes(trafficSim.status) ? [new ScatterplotLayer({
      id: 'sim-vehicles',
      data: { length: simFrame.count, attributes: { getPosition: { value: simFrame.data, size: 2, stride: 16 } } },
      coordinateSystem: COORDINATE_SYSTEM.METER_OFFSETS,
      coordinateOrigin: [simFrame.origin.lon0, simFrame.origin.lat0, 0],
      getRadius: 2.5, radiusUnits: 'meters', radiusMinPixels: 3,
      getFillColor: [255, 60, 90, 255], pickable: false,
    })] : []),
    ...(editor.selected ? [new GeoJsonLayer({ id: 'selected-feature',
      data: { type: 'FeatureCollection', features: [editor.selected] }, filled: false, stroked: true,
      getLineColor: [255, 255, 255], getLineWidth: 3, lineWidthUnits: 'pixels', pickable: false,
    })] : []),
    new GeoJsonLayer({ id: 'draft-feature', data: draftCollection, pointType: 'circle',
      getPointRadius: 5, pointRadiusUnits: 'pixels', getFillColor: [255, 220, 50],
      getLineColor: [255, 220, 50], lineWidthMinPixels: 3, pickable: false }),
    new ScatterplotLayer({ id: 'editable-vertices', data: editVertices,
      getPosition: d => d.position, getRadius: 5, radiusUnits: 'pixels',
      getFillColor: [255, 255, 255], getLineColor: [20, 30, 40], stroked: true, lineWidthMinPixels: 1, pickable: true }),
  ], [roadFeatures, buildingCollection, treeCollection, carCollection, updateHoverTooltip, editor.selected, editVertices, draftCollection, simFrame, trafficSim.status]);

  return (
    <div className="hud-font" style={{ width: '100vw', height: '100vh', position: 'relative', backgroundColor: '#050914', overflow: 'hidden' }}>
      
      {/* ── CSS Stilleri ── */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&display=swap');
        
        .hud-font { font-family: 'Rajdhani', sans-serif; }
        .mono-font { font-family: 'Share Tech Mono', monospace; }
        
        .hud-panel {
          background: rgba(10, 15, 25, 0.78);
          backdrop-filter: blur(12px);
          border: 1px solid rgba(0, 255, 157, 0.25);
          box-shadow: 0 0 20px rgba(0, 255, 157, 0.08), inset 0 0 20px rgba(0, 255, 157, 0.05);
          color: #e2e8f0;
          border-radius: 4px;
        }
        .hud-panel-right {
          border: 1px solid rgba(0, 255, 157, 0.25);
          box-shadow: 0 0 20px rgba(0, 255, 157, 0.08), inset 0 0 20px rgba(0, 255, 157, 0.05);
        }
        
        .hud-title {
          color: #00FF9D;
          text-transform: uppercase;
          letter-spacing: 2px;
          font-weight: 700;
          font-size: 14px;
          border-bottom: 1px solid rgba(0, 255, 157, 0.3);
          padding-bottom: 8px;
          margin-bottom: 14px;
          text-shadow: 0 0 8px rgba(0, 255, 157, 0.4);
        }
        .hud-title-green {
          color: #00FF9D;
          border-bottom: 1px solid rgba(0, 255, 157, 0.3);
          text-shadow: 0 0 8px rgba(0, 255, 157, 0.4);
        }
        
        .cyber-button {
          background: rgba(0, 255, 157, 0.1);
          border: 1px solid #00FF9D;
          color: #00FF9D;
          text-transform: uppercase;
          letter-spacing: 2px;
          font-weight: 700;
          padding: 10px 18px;
          cursor: pointer;
          transition: all 0.2s;
          box-shadow: 0 0 15px rgba(0, 255, 157, 0.15);
          border-radius: 2px;
        }
        .cyber-button:hover:not(:disabled) {
          background: rgba(0, 255, 157, 0.25);
          box-shadow: 0 0 25px rgba(0, 255, 157, 0.4);
          text-shadow: 0 0 5px #00FF9D;
        }
        .cyber-button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
          border-color: rgba(0, 255, 157, 0.3);
          color: rgba(0, 255, 157, 0.5);
        }

        .search-result-item {
          padding: 8px 10px;
          border-bottom: 1px solid rgba(0, 255, 157, 0.15);
          cursor: pointer;
          transition: background 0.15s;
        }
        .search-result-item:hover {
          background: rgba(0, 255, 157, 0.2);
        }

        /* Harita kontrollerini gizle */
        .mapboxgl-ctrl-bottom-left, .mapboxgl-ctrl-bottom-right { display: none !important; }
      `}</style>

      {/* ── Arka Plan: Harita ── */}
      <div style={{ position: 'absolute', inset: 0, zIndex: 1 }}>
        {!MAPBOX_TOKEN.startsWith('TODO') ? (
          <DeckGL
            viewState={viewState}
            onViewStateChange={({ viewState: vs }) => setViewState(vs)}
            controller={{ doubleClickZoom: editor.mode === 'select' }}
            layers={layers}
            onClick={editor.onMapClick}
            getCursor={() => editor.mode === 'select' ? 'grab' : 'crosshair'}
            getTooltip={null}
          >
            <Map
              mapboxAccessToken={MAPBOX_TOKEN}
              mapStyle={basemap === 'esri' ? ESRI_SATELLITE_STYLE : 'mapbox://styles/mapbox/satellite-v9'}
              reuseMaps
              onError={(e) => {
                // Uydu karoları reddedilirse harita eski düşük zoom karolarını büyütüp pikselli gösterir;
                // bu durumda Esri altlığına geç.
                const status = e?.error?.status;
                if (basemap === 'mapbox' && (status === 429 || status === 401 || status === 403)) {
                  console.warn(`Mapbox uydu karoları reddedildi (HTTP ${status}), Esri altlığına geçiliyor.`);
                  setBasemap('esri');
                }
              }}
            />
          </DeckGL>
        ) : (
          <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#00FF9D' }}>
            MAPBOX TOKEN EKSİK
          </div>
        )}
      </div>

      {/* ── ÜST BAR ── */}
      <div className="hud-panel" style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '44px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0 30px', zIndex: 10, borderTop: 'none', borderLeft: 'none', borderRight: 'none', borderRadius: 0 }}>
        <div style={{ fontSize: '20px', fontWeight: 700, letterSpacing: '3px', color: '#00FF9D', display: 'flex', alignItems: 'center', gap: '15px' }}>
          <div style={{ width: '12px', height: '12px', background: '#00FF9D', boxShadow: '0 0 10px #00FF9D' }}></div>
          <span><span style={{color: '#fff'}}>GEO</span>SIM <span style={{ opacity: 0.5 }}>//</span> DİJİTAL İKİZ KOMUTA MERKEZİ</span>
          {activeRegionName && (
            <span className="mono-font" style={{ fontSize: '12px', color: '#ff8c00', border: '1px solid rgba(255,140,0,0.4)', padding: '2px 8px', borderRadius: '2px' }}>
              HEDEF: {activeRegionName}
            </span>
          )}
        </div>
        <div className="mono-font" style={{ fontSize: '15px', color: '#00FF9D', letterSpacing: '1px', textShadow: '0 0 8px #00FF9D' }}>
          <LiveClock />
        </div>
      </div>

      {/* ── SOL PANEL: Analiz ── */}
      <div className="hud-panel" style={{ position: 'absolute', top: '64px', left: '20px', width: '330px', bottom: '20px', padding: '20px', zIndex: 10, display: 'flex', flexDirection: 'column', overflowY: 'auto' }}>
        <div className="hud-title">BÖLGE & KONUM ANALİZİ</div>

        {/* ── 1. Bölge / Konum Arama Kutusu ve ARA Butonu ── */}
        <div ref={searchBoxRef} style={{ position: 'relative', marginBottom: '16px' }}>
          <div style={{ fontSize: '11px', color: '#8ba4b8', marginBottom: '6px', letterSpacing: '1px' }}>KONUM / BÖLGE ARA</div>
          <form onSubmit={handleSearchSubmit} style={{ display: 'flex', gap: '6px' }}>
            <input
              type="text"
              value={searchQuery}
              onFocus={() => { if (searchResults.length > 0) setShowSearchDropdown(true); }}
              onChange={e => setSearchQuery(e.target.value)}
              placeholder="İlçe, mahalle, cadde veya yer yazın..."
              className="mono-font"
              style={{
                flex: 1,
                background: 'rgba(0, 0, 0, 0.65)',
                border: '1px solid rgba(0, 255, 157, 0.35)',
                color: '#00FF9D',
                padding: '8px 10px',
                fontSize: '12px',
                outline: 'none',
                borderRadius: '2px',
              }}
            />
            <button
              type="submit"
              disabled={isSearching}
              className="mono-font"
              style={{
                background: 'rgba(0, 255, 157, 0.18)',
                border: '1px solid #00FF9D',
                color: '#00FF9D',
                padding: '8px 14px',
                fontWeight: 700,
                fontSize: '12px',
                cursor: 'pointer',
                letterSpacing: '1px',
                borderRadius: '2px',
                boxShadow: '0 0 10px rgba(0, 255, 157, 0.2)',
              }}
            >
              {isSearching ? '...' : 'ARA'}
            </button>
          </form>

          {/* Açılır Arama Sonuçları Listesi */}
          {showSearchDropdown && searchResults.length > 0 && (
            <div
              style={{
                position: 'absolute',
                top: '100%',
                left: 0,
                right: 0,
                marginTop: '4px',
                background: 'rgba(6, 11, 22, 0.98)',
                border: '1px solid #00FF9D',
                boxShadow: '0 8px 25px rgba(0, 0, 0, 0.85), 0 0 15px rgba(0, 255, 157, 0.25)',
                borderRadius: '3px',
                zIndex: 50,
                maxHeight: '230px',
                overflowY: 'auto',
              }}
            >
              {searchResults.map((item) => (
                <div
                  key={item.id}
                  className="search-result-item"
                  onClick={() => handleSelectSearchResult(item)}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span className="mono-font" style={{ color: '#00FF9D', fontSize: '12px', fontWeight: 700 }}>
                      {item.name}
                    </span>
                    <span className="mono-font" style={{ fontSize: '10px', color: '#ff8c00' }}>
                      SEÇ & TARA ▸
                    </span>
                  </div>
                  <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '2px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {item.full_name}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* ── 2. Koordinat Girişi ve Koordinata Git & Analiz Et ── */}
        <div style={{ marginBottom: '16px' }}>
          <div style={{ fontSize: '11px', color: '#8ba4b8', marginBottom: '6px', letterSpacing: '1px' }}>KOORDİNAT BİLGİSİ (ENLEM / BOYLAM)</div>
          <div className="mono-font" style={{ background: 'rgba(0,0,0,0.6)', padding: '10px 12px', border: '1px solid rgba(0, 255, 157, 0.2)', color: '#00FF9D', fontSize: '13px', lineHeight: 1.5 }}>
            <div style={{ display: 'flex', alignItems: 'center', marginBottom: '6px' }}>
              <span style={{ width: '42px', color: '#8ba4b8' }}>LAT:</span>
              <input
                type="text"
                inputMode="decimal"
                value={inputLat}
                onFocus={() => { isEditingCoordsRef.current = true; }}
                onBlur={() => { isEditingCoordsRef.current = false; }}
                onChange={e => handleCoordInputChange('lat', e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter') handleJump(); }}
                placeholder="41.008200"
                style={{ background: 'rgba(0,255,157,0.04)', border: 'none', borderBottom: '1px solid rgba(0,255,157,0.35)', color: '#00FF9D', outline: 'none', width: '100%', padding: '3px 6px', fontFamily: 'inherit', fontSize: '13px' }}
              />
            </div>
            <div style={{ display: 'flex', alignItems: 'center', marginBottom: '10px' }}>
              <span style={{ width: '42px', color: '#8ba4b8' }}>LON:</span>
              <input
                type="text"
                inputMode="decimal"
                value={inputLon}
                onFocus={() => { isEditingCoordsRef.current = true; }}
                onBlur={() => { isEditingCoordsRef.current = false; }}
                onChange={e => handleCoordInputChange('lon', e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter') handleJump(); }}
                placeholder="28.978400"
                style={{ background: 'rgba(0,255,157,0.04)', border: 'none', borderBottom: '1px solid rgba(0,255,157,0.35)', color: '#00FF9D', outline: 'none', width: '100%', padding: '3px 6px', fontFamily: 'inherit', fontSize: '13px' }}
              />
            </div>
            <button
              onClick={handleJump}
              disabled={isAnalyzing}
              style={{
                background: 'rgba(0,255,157,0.14)',
                border: '1px solid #00FF9D',
                color: '#00FF9D',
                width: '100%',
                padding: '7px',
                cursor: isAnalyzing ? 'not-allowed' : 'pointer',
                transition: 'all 0.2s',
                fontWeight: 'bold',
                fontSize: '12px',
                letterSpacing: '1px',
              }}
              onMouseOver={e => { if (!isAnalyzing) e.currentTarget.style.background = 'rgba(0,255,157,0.3)'; }}
              onMouseOut={e => { e.currentTarget.style.background = 'rgba(0,255,157,0.14)'; }}
            >
              KOORDİNATA GİT & ANALİZ ET
            </button>
          </div>
        </div>

        {/* Analiz Durumu */}
        <div style={{ marginBottom: '12px', background: 'rgba(0, 255, 157, 0.05)', border: '1px solid rgba(0, 255, 157, 0.2)', padding: '8px 12px', borderRadius: '4px' }}>
          <div style={{ fontSize: '10px', color: '#8ba4b8', letterSpacing: '1px' }}>BİNA TESPİT YÖNTEMİ</div>
          <div className="mono-font" style={{ fontSize: '12px', color: '#00FF9D', fontWeight: 700, marginTop: '2px' }}>
            OSM + UYDU GÖRÜNTÜSÜNDE YAPAY ZEKÂ
          </div>
          <div className="mono-font" style={{ fontSize: '11px', color: isAiEnriching ? '#fbbf24' : '#8ba4b8', marginTop: '2px' }}>
            {isAiEnriching ? 'Uydu görüntüsü taranıyor…' : editor.region ? `Kayıtlı alan: ${editor.region.name}` : 'Hazır'}
          </div>
        </div>

        {/* Tarama Butonu */}
        <button className="cyber-button mono-font" onClick={() => handleAnalyze()} disabled={isAnalyzing} style={{ width: '100%', marginBottom: '8px' }}>
          {isAnalyzing ? 'TARANIYOR...' : 'BU BÖLGEYİ ANALİZ ET'}
        </button>

        {/* OSM Overpass Yol Ağı Çekme Butonu */}
        <button
          className="mono-font"
          onClick={() => fetchRoads()}
          disabled={isLoadingRoads}
          style={{
            width: '100%',
            marginBottom: '14px',
            background: 'rgba(255, 140, 0, 0.12)',
            border: '1px solid #ff8c00',
            color: '#ff8c00',
            textTransform: 'uppercase',
            letterSpacing: '1.5px',
            fontWeight: 700,
            fontSize: '12px',
            padding: '7px 14px',
            cursor: isLoadingRoads ? 'not-allowed' : 'pointer',
            borderRadius: '2px',
            boxShadow: '0 0 12px rgba(255, 140, 0, 0.18)',
            transition: 'all 0.2s',
          }}
        >
          {isLoadingRoads ? 'YOL AĞI ÇEKİLİYOR...' : `YOL AĞINI ÇEK (OSM • ${counts.road})`}
        </button>

        {/* Hızlı Test Bölgeleri */}
        <div style={{ marginBottom: '14px' }}>
          <div style={{ fontSize: '11px', color: '#8ba4b8', marginBottom: '6px', letterSpacing: '1px' }}>HAZIR TEST BÖLGELERİ</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
            {PRESET_LOCATIONS.map((loc, idx) => (
              <button
                key={idx}
                onClick={() => {
                  setActiveRegionName(loc.name);
                  setInputLat(loc.lat.toFixed(6));
                  setInputLon(loc.lon.toFixed(6));
                  handleAnalyze({ lat: loc.lat, lon: loc.lon, zoom: loc.zoom });
                }}
                disabled={isAnalyzing}
                style={{
                  background: 'rgba(0, 255, 157, 0.08)',
                  border: '1px solid rgba(0, 255, 157, 0.3)',
                  color: '#e2e8f0',
                  padding: '6px 8px',
                  borderRadius: '3px',
                  cursor: 'pointer',
                  textAlign: 'left',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  fontSize: '11px',
                  transition: 'all 0.2s',
                }}
                onMouseOver={e => e.currentTarget.style.background = 'rgba(0, 255, 157, 0.2)'}
                onMouseOut={e => e.currentTarget.style.background = 'rgba(0, 255, 157, 0.08)'}
              >
                <span className="mono-font" style={{ color: '#00FF9D', fontWeight: 700 }}>{loc.name}</span>
                <span style={{ fontSize: '10px', color: '#8ba4b8' }}>{loc.desc}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Hata */}
        {error && (
          <div style={{ color: '#f43f5e', fontSize: '12px', marginBottom: '12px', borderLeft: '2px solid #f43f5e', paddingLeft: '8px' }}>
            {error}
          </div>
        )}

        {/* Metrikler */}
        <div className="hud-title" style={{ marginTop: 'auto' }}>TESPİT METRİKLERİ</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '6px' }}>
          <MetricBox label="BİNA" val={counts.building} color="#00FF9D" />
          <MetricBox label="OSM"  val={counts.osm}      color={toCss(SOURCE_STYLES.osm.rgb)} />
          <MetricBox label="YZ"   val={counts.ai}       color={toCss(SOURCE_STYLES.ai.rgb)} />
          <MetricBox label="YOL"  val={counts.road}     color={toCss(ROAD_RGB)} />
        </div>
      </div>

      <Legend />

      {/* ── SAĞ PANEL: Düzenleme ── */}
      <div className="hud-panel hud-panel-right" style={{ position: 'absolute', top: '64px', right: '20px', width: '320px', bottom: '20px', padding: '24px', zIndex: 10, overflowY: 'auto' }}>
        <div className="hud-title hud-title-green">HARİTA DÜZENLEME</div>
        <RegionEditor editor={editor} collection={geoJson} tooltip={tooltip} />
      </div>

      <SimulationPanel sim={trafficSim} hasData={roadFeatures.length > 1} />

    </div>
  );
}
