/**
 * App.jsx — GeoSim Ana Uygulama
 * Yeni mimari: Tam ekran Mapbox + Deck.gl 3D haritası
 */

import React from 'react';
import DeckMapView from './components/map/DeckMapView';

function App() {
  return (
    <div style={{ width: '100vw', height: '100vh', overflow: 'hidden', background: '#0f172a' }}>
      <DeckMapView />
    </div>
  );
}

export default App;
