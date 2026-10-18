/**
 * DeckMapView.jsx — Mapbox + Deck.gl 3D Kentsel Simülasyon Haritası
 * Fütüristik "Akıllı Şehir Komuta Merkezi" (Smart City HUD) Konsepti
 */

import React, { useState, useCallback, useEffect } from 'react';
import Map from 'react-map-gl/mapbox';
import DeckGL from '@deck.gl/react';
import { GeoJsonLayer, PathLayer } from '@deck.gl/layers';
import { FlyToInterpolator } from '@deck.gl/core';
import 'mapbox-gl/dist/mapbox-gl.css';

const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN || 'TODO_MAPBOX_TOKEN_BURAYA';
const BACKEND_URL  = 'http://localhost:8000';

const INITIAL_VIEW = {
  longitude:  28.9784,
  latitude:   41.0082,
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
    <div style={{ background: 'rgba(0,0,0,0.5)', border: `1px solid ${color}40`, padding: '12px 8px', textAlign: 'center', borderRadius: '4px' }}>
      <div className="mono-font" style={{ fontSize: '28px', fontWeight: 700, color: color, textShadow: `0 0 15px ${color}80`, lineHeight: 1 }}>
        {val}
      </div>
      <div style={{ fontSize: '11px', color: '#8ba4b8', marginTop: '8px', letterSpacing: '1px' }}>{label}</div>
    </div>
  );
}

const PRESET_LOCATIONS = [
  { name: 'FATİH / BALAT', desc: '66 Bina • %97 Güven', lat: 41.0295, lon: 28.9485, zoom: 17.0 },
  { name: 'KADIKÖY / MODA', desc: '68 Bina • %98.7 Güven', lat: 40.9855, lon: 29.0275, zoom: 17.0 },
  { name: 'SULTANAHMET', desc: '27 Bina • %96.5 Güven', lat: 41.0055, lon: 28.9765, zoom: 17.0 },
];

// Her bina örneği için ayırt edilebilir renk (golden-angle HSL)
function instanceColor(i) {
  const h = (i * 137.508) % 360, s = 0.85, l = 0.55;
  const k = n => (n + h / 30) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = n => Math.round(255 * (l - a * Math.max(-1, Math.min(k(n) - 3, 9 - k(n), 1))));
  return [f(0), f(8), f(4)];
}

// ── Ana Bileşen ───────────────────────────────────────────────────────────────
export default function DeckMapView() {
  const [viewState,      setViewState]      = useState(INITIAL_VIEW);
  const [geoJson,        setGeoJson]        = useState(null);
  const [roadsGeoJson,   setRoadsGeoJson]   = useState(null);
  const [isAnalyzing,    setIsAnalyzing]    = useState(false);
  const [isLoadingRoads, setIsLoadingRoads] = useState(false);
  const [error,          setError]          = useState('');
  const [tooltip,        setTooltip]        = useState(null);
  const [inputLat,       setInputLat]       = useState(INITIAL_VIEW.latitude.toFixed(6));
  const [inputLon,       setInputLon]       = useState(INITIAL_VIEW.longitude.toFixed(6));

  useEffect(() => {
    setInputLat(viewState.latitude.toFixed(6));
    setInputLon(viewState.longitude.toFixed(6));
  }, [viewState.latitude, viewState.longitude]);

  // ── Overpass API Yol Ağlarını Çek (/api/roads) ─────────────────────────────
  const fetchRoads = useCallback(async (customCoords = null) => {
    setIsLoadingRoads(true);
    const targetLat  = customCoords?.lat  ?? parseFloat(inputLat) ?? viewState.latitude;
    const targetLon  = customCoords?.lon  ?? parseFloat(inputLon) ?? viewState.longitude;
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
    } catch (err) {
      console.error('Yol ağı çekme hatası:', err);
    } finally {
      setIsLoadingRoads(false);
    }
  }, [viewState.latitude, viewState.longitude, viewState.zoom, inputLat, inputLon]);

  // İlk açılışta mevcut bölgenin yol ağını otomatik çek
  useEffect(() => {
    fetchRoads({
      lat:  INITIAL_VIEW.latitude,
      lon:  INITIAL_VIEW.longitude,
      zoom: INITIAL_VIEW.zoom,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Analiz İste (Otomatik Zoom Koruması + Paralel Yol Çekimi ile) ──────────
  const handleAnalyze = useCallback(async (customCoords = null) => {
    setIsAnalyzing(true);
    setError('');

    const targetLat  = customCoords?.lat  ?? parseFloat(inputLat) ?? viewState.latitude;
    const targetLon  = customCoords?.lon  ?? parseFloat(inputLon) ?? viewState.longitude;
    // Binaların yüksek doğrulukla tespit edilmesi için en az 16.8 zoom gereklidir
    const targetZoom = customCoords?.zoom ?? (viewState.zoom < 16.0 ? 17.0 : viewState.zoom);

    // Eğer harita çok uzaktaysa kamerayı otomatik olarak binaya yaklaştır
    if (viewState.zoom < 16.0 || customCoords) {
      setViewState(prev => ({
        ...prev,
        latitude: targetLat,
        longitude: targetLon,
        zoom: targetZoom,
        transitionDuration: 1200,
        transitionInterpolator: new FlyToInterpolator()
      }));
    }

    // Overpass API üzerinden yol ağlarını paralel olarak çek
    fetchRoads({ lat: targetLat, lon: targetLon, zoom: targetZoom });

    try {
      const response = await fetch(`${BACKEND_URL}/analyze`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({
          latitude:  targetLat,
          longitude: targetLon,
          zoom:      targetZoom,
        }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Sunucu hatası: ${response.status}`);
      }

      const data = await response.json();
      setGeoJson(data);
    } catch (err) {
      console.error('Analiz hatası:', err);
      setError(err.message || 'Bağlantı hatası.');
    } finally {
      setIsAnalyzing(false);
    }
  }, [viewState, inputLat, inputLon, fetchRoads]);

  const handleJump = () => {
    const lat = parseFloat(inputLat) || viewState.latitude;
    const lon = parseFloat(inputLon) || viewState.longitude;
    setViewState(prev => ({
      ...prev,
      latitude: lat,
      longitude: lon,
      zoom: 17.0,
      transitionDuration: 1500,
      transitionInterpolator: new FlyToInterpolator()
    }));
    handleAnalyze({ lat, lon, zoom: 17.0 });
  };
  
  // Yol verilerini birleştir (/api/roads üzerinden gelen veya /analyze içinde dönen)
  const roadFeatures = (roadsGeoJson?.features?.length > 0)
    ? roadsGeoJson.features
    : (geoJson?.features?.filter(f => f.properties?.type === 'road') ?? []);

  // HUD için verileri hesapla
  const counts = { building: 0, road: roadFeatures.length, tree: 0, car: 0 };
  if (geoJson?.features) {
    geoJson.features.forEach(f => {
      const t = f.properties?.type;
      if (t && t !== 'road' && counts[t] !== undefined) counts[t]++;
    });
  }

  // ── Deck.gl Katmanları (Hibrit Mimari: Yol Ağı Altta, Bina Katmanı Üstte) ──
  const layers = [
    // 1. Yol Ağı Neon Glow Alt Katmanı (parlama efekti için yarı saydam dış hale)
    ...(roadFeatures.length > 0 ? [
      new PathLayer({
        id:             'osm-roads-glow-layer',
        data:           roadFeatures,
        getPath:        d => d.geometry.coordinates,
        getColor:       [255, 140, 0, 75],
        getWidth:       d => (d.properties?.width ?? 3) + 5,
        widthUnits:     'pixels',
        widthMinPixels: 5,
        widthMaxPixels: 18,
        capRounded:     true,
        jointRounded:   true,
        pickable:       false,
        updateTriggers: { data: roadFeatures },
      }),
      // 2. Ana Hazır Yol Ağı Katmanı (bina katmanının hemen altında, neon turuncu PathLayer)
      new PathLayer({
        id:             'osm-roads-layer',
        data:           roadFeatures,
        getPath:        d => d.geometry.coordinates,
        getColor:       [255, 140, 0],
        getWidth:       3,
        widthUnits:     'pixels',
        widthMinPixels: 2,
        widthMaxPixels: 10,
        capRounded:     true,
        jointRounded:   true,
        pickable:       true,
        autoHighlight:  true,
        highlightColor: [0, 255, 255, 230],
        onHover:        info => setTooltip(info.object ? {
          text: `YOL AĞI // ${info.object.properties?.name || 'SOKAK/CADDE'} [${(info.object.properties?.highway || info.object.properties?.class || 'highway').toUpperCase()}]`,
        } : null),
        updateTriggers: { data: roadFeatures },
      }),
    ] : []),

    // 3. Binalar — 2D Instance Segmentation maskeleri (her bina farklı renk, yolların üstünde)
    ...(geoJson ? [
      new GeoJsonLayer({
        id:                 'buildings-layer',
        data:               {
          type:     'FeatureCollection',
          features: (geoJson.features?.filter(f => f.properties?.type === 'building') ?? [])
            .map((f, i) => ({ ...f, properties: { ...f.properties, _idx: i } })),
        },
        stroked:            true,
        filled:             true,
        extruded:           false,
        getFillColor:       f => [...instanceColor(f.properties._idx), 90],
        getLineColor:       f => [...instanceColor(f.properties._idx), 255],
        lineWidthMinPixels: 2,
        pickable:           true,
        autoHighlight:      true,
        highlightColor:     [255, 255, 255, 110],
        onHover:            info => setTooltip(info.object ? {
          text: `BİNA #${info.object.properties._idx + 1} // GÜVEN: %${Math.round((info.object.properties?.confidence ?? 1) * 100)}`,
        } : null),
        updateTriggers:     { data: geoJson },
      }),

      // 4. Ağaçlar — Point
      new GeoJsonLayer({
        id:                'trees-layer',
        data:              {
          type:     'FeatureCollection',
          features: geoJson.features?.filter(f => f.properties?.type === 'tree') ?? [],
        },
        stroked:           true,
        filled:            true,
        pointType:         'circle',
        getPointRadius:    4,
        pointRadiusUnits:  'pixels',
        getFillColor:      [0, 255, 157, 60],
        getLineColor:      [0, 255, 157, 255],
        lineWidthMinPixels: 1.5,
        pickable:          true,
        onHover:           info => setTooltip(info.object ? {
          text: 'BİTKİ ÖRTÜSÜ',
        } : null),
        updateTriggers:    { data: geoJson },
      }),

      // 5. Araçlar — Point
      new GeoJsonLayer({
        id:                'cars-layer',
        data:              {
          type:     'FeatureCollection',
          features: geoJson.features?.filter(f => f.properties?.type === 'car') ?? [],
        },
        stroked:           true,
        filled:            true,
        pointType:         'circle',
        getPointRadius:    3.5,
        pointRadiusUnits:  'pixels',
        getFillColor:      [255, 0, 85, 120], // Neon pembe/kırmızı
        getLineColor:      [255, 0, 85, 255],
        lineWidthMinPixels: 1.5,
        pickable:          true,
        onHover:           info => setTooltip(info.object ? {
          text: 'HAREKETLİ OBJE // ARAÇ',
        } : null),
        updateTriggers:    { data: geoJson },
      }),
    ] : []),
  ];

  return (
    <div className="hud-font" style={{ width: '100vw', height: '100vh', position: 'relative', backgroundColor: '#050914', overflow: 'hidden' }}>
      
      {/* ── CSS Stilleri ── */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&display=swap');
        
        .hud-font { font-family: 'Rajdhani', sans-serif; }
        .mono-font { font-family: 'Share Tech Mono', monospace; }
        
        .hud-panel {
          background: rgba(10, 15, 25, 0.75);
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
          margin-bottom: 16px;
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
          padding: 12px 20px;
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

        /* Scanline efekti (hafif tarama çizgileri) */
        .scanlines {
          position: absolute; inset: 0; z-index: 5; pointer-events: none;
          background: linear-gradient(to bottom, rgba(255,255,255,0), rgba(255,255,255,0) 50%, rgba(0,0,0,0.1) 50%, rgba(0,0,0,0.1));
          background-size: 100% 4px;
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
            controller={true}
            layers={layers}
            getTooltip={null}
          >
            <Map
              mapboxAccessToken={MAPBOX_TOKEN}
              mapStyle="mapbox://styles/mapbox/satellite-v9"
              reuseMaps
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
        </div>
        <div className="mono-font" style={{ fontSize: '15px', color: '#00FF9D', letterSpacing: '1px', textShadow: '0 0 8px #00FF9D' }}>
          <LiveClock />
        </div>
      </div>

      {/* ── SOL PANEL: Analiz ── */}
      <div className="hud-panel" style={{ position: 'absolute', top: '64px', left: '20px', width: '320px', bottom: '20px', padding: '24px', zIndex: 10, display: 'flex', flexDirection: 'column' }}>
        <div className="hud-title">BÖLGE ANALİZİ</div>
        
        {/* Koordinatlar */}
        <div style={{ marginBottom: '25px' }}>
          <div style={{ fontSize: '12px', color: '#8ba4b8', marginBottom: '6px', letterSpacing: '1px' }}>KOORDİNAT BİLGİSİ</div>
          <div className="mono-font" style={{ background: 'rgba(0,0,0,0.6)', padding: '12px', border: '1px solid rgba(0, 255, 157, 0.2)', color: '#00FF9D', fontSize: '14px', lineHeight: 1.6 }}>
            <div style={{ display: 'flex', alignItems: 'center', marginBottom: '4px' }}>
              <span style={{ width: '40px' }}>LAT:</span>
              <input type="number" step="0.0001" value={inputLat} onChange={e => setInputLat(e.target.value)} style={{ background: 'transparent', border: 'none', borderBottom: '1px solid rgba(0,255,157,0.3)', color: '#00FF9D', outline: 'none', width: '100%', padding: '2px 4px' }} />
            </div>
            <div style={{ display: 'flex', alignItems: 'center', marginBottom: '12px' }}>
              <span style={{ width: '40px' }}>LON:</span>
              <input type="number" step="0.0001" value={inputLon} onChange={e => setInputLon(e.target.value)} style={{ background: 'transparent', border: 'none', borderBottom: '1px solid rgba(0,255,157,0.3)', color: '#00FF9D', outline: 'none', width: '100%', padding: '2px 4px' }} />
            </div>
            <button onClick={handleJump} style={{ background: 'rgba(0,255,157,0.1)', border: '1px solid #00FF9D', color: '#00FF9D', width: '100%', padding: '6px', cursor: 'pointer', transition: 'all 0.2s', fontWeight: 'bold', fontSize: '12px' }} onMouseOver={e => e.target.style.background='rgba(0,255,157,0.3)'} onMouseOut={e => e.target.style.background='rgba(0,255,157,0.1)'}>
              KOORDİNATA GİT
            </button>
          </div>
        </div>

        {/* Model Bilgisi */}
        <div style={{ marginBottom: '16px', background: 'rgba(0, 255, 157, 0.05)', border: '1px solid rgba(0, 255, 157, 0.2)', padding: '10px 12px', borderRadius: '4px' }}>
          <div style={{ fontSize: '10px', color: '#8ba4b8', letterSpacing: '1px' }}>AKTİF SEGMENTASYON MOTORU</div>
          <div className="mono-font" style={{ fontSize: '12px', color: '#00FF9D', fontWeight: 700, marginTop: '3px' }}>
            ROBOFLOW // 9.665 GÖRSEL
          </div>
          <div className="mono-font" style={{ fontSize: '10px', color: '#64748b', marginTop: '1px' }}>
            buildings-instance-segmentation/3
          </div>
        </div>

        {/* Tarama Butonu */}
        <button className="cyber-button mono-font" onClick={() => handleAnalyze()} disabled={isAnalyzing} style={{ width: '100%', marginBottom: '8px' }}>
          {isAnalyzing ? 'TARANIYOR...' : (geoJson ? 'YENİDEN TARA' : 'BÖLGEYİ TARA')}
        </button>

        {/* OSM Overpass Yol Ağı Çekme Butonu */}
        <button
          className="mono-font"
          onClick={() => fetchRoads()}
          disabled={isLoadingRoads}
          style={{
            width: '100%',
            marginBottom: '16px',
            background: 'rgba(255, 140, 0, 0.12)',
            border: '1px solid #ff8c00',
            color: '#ff8c00',
            textTransform: 'uppercase',
            letterSpacing: '1.5px',
            fontWeight: 700,
            fontSize: '12px',
            padding: '8px 14px',
            cursor: isLoadingRoads ? 'not-allowed' : 'pointer',
            borderRadius: '2px',
            boxShadow: '0 0 12px rgba(255, 140, 0, 0.18)',
            transition: 'all 0.2s',
          }}
        >
          {isLoadingRoads ? 'YOL AĞI ÇEKİLİYOR...' : `YOL AĞINI ÇEK (OSM • ${counts.road})`}
        </button>

        {/* Hızlı Test Bölgeleri */}
        <div style={{ marginBottom: '16px' }}>
          <div style={{ fontSize: '11px', color: '#8ba4b8', marginBottom: '6px', letterSpacing: '1px' }}>HAZIR TEST BÖLGELERİ</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            {PRESET_LOCATIONS.map((loc, idx) => (
              <button
                key={idx}
                onClick={() => {
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
          <div style={{ color: '#f43f5e', fontSize: '12px', marginBottom: '16px', borderLeft: '2px solid #f43f5e', paddingLeft: '8px' }}>
            HATA: {error}
          </div>
        )}

        {/* Metrikler */}
        <div className="hud-title" style={{ marginTop: 'auto' }}>TESPİT METRİKLERİ</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
          <MetricBox label="BİNA" val={counts.building} color="#00FF9D" />
          <MetricBox label="YOL"  val={counts.road}     color="#f59e0b" />
          <MetricBox label="AĞAÇ" val={counts.tree}     color="#00FF9D" />
          <MetricBox label="ARAÇ" val={counts.car}      color="#f43f5e" />
        </div>
      </div>

      {/* ── SAĞ PANEL: Düzenleme ── */}
      <div className="hud-panel hud-panel-right" style={{ position: 'absolute', top: '64px', right: '20px', width: '320px', bottom: '20px', padding: '24px', zIndex: 10 }}>
        <div className="hud-title hud-title-green">DÜZENLEME & DENETİM</div>
        
        {tooltip ? (
           <div>
             <div style={{ fontSize: '12px', color: '#8ba4b8', marginBottom: '10px', letterSpacing: '1px' }}>SEÇİLİ OBJE DETAYI</div>
             <div className="mono-font" style={{ background: 'rgba(0, 255, 157, 0.08)', padding: '16px', borderLeft: '3px solid #00FF9D', color: '#fff', fontSize: '15px' }}>
               {tooltip.text}
             </div>
             
             {/* Sahte Kontroller */}
             <div style={{ marginTop: '30px' }}>
               <div style={{ fontSize: '11px', color: '#8ba4b8', marginBottom: '8px', letterSpacing: '1px' }}>YAPISAL YÜKSEKLİK (M)</div>
               <input type="range" min="0" max="100" defaultValue="15" style={{ width: '100%', accentColor: '#00FF9D', cursor: 'pointer' }}/>
             </div>
             <div style={{ marginTop: '20px' }}>
               <div style={{ fontSize: '11px', color: '#8ba4b8', marginBottom: '8px', letterSpacing: '1px' }}>SİSTEM ONAYI</div>
               <button style={{ background: 'transparent', border: '1px solid #f43f5e', color: '#f43f5e', padding: '6px 12px', fontSize: '12px', cursor: 'pointer', fontFamily: 'inherit' }}>
                 OBJEYİ İMHA ET
               </button>
             </div>
           </div>
        ) : (
           <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '60%', opacity: 0.5 }}>
             <div style={{ width: '40px', height: '40px', border: '1px dashed #00FF9D', borderRadius: '50%', animation: 'spin 4s linear infinite', marginBottom: '15px' }}></div>
             <div style={{ color: '#00FF9D', fontSize: '13px', letterSpacing: '1px' }}>HEDEF BEKLENİYOR...</div>
           </div>
        )}
      </div>



    </div>
  );
}
