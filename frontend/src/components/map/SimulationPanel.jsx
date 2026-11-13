import React, { useState } from 'react';
import { DENSITIES } from '../../simulation/engine.js';
import { SEEDS } from '../../simulation/rng.js';

const LABEL = { fontSize: 11, opacity: 0.7, letterSpacing: 1, textTransform: 'uppercase' };
const FIELD = { width: '100%', background: 'rgba(0,0,0,0.35)', color: '#e2e8f0', border: '1px solid rgba(0,255,157,0.3)', padding: '5px 6px', fontSize: 13 };

export default function SimulationPanel({ sim, hasData }) {
  const [open, setOpen] = useState(false);
  const { status, report, live, error, settings, configure, play, pause, reset } = sim;
  const running = status === 'playing';
  const scenarios = report?.scenarios ?? {};
  const closure = scenarios.S1?.closure;

  return (
    <div className="hud-panel mono-font" style={{ position: 'absolute', left: 370, bottom: 20, zIndex: 10, width: open ? 360 : 'auto', padding: open ? 14 : 0 }}>
      {!open ? (
        <button className="cyber-button" disabled={!hasData} onClick={() => setOpen(true)}
          title={hasData ? '' : 'Önce bir bölgeyi analiz edin (yol ağı gerekir)'}>TRAFİK SİMÜLASYONU</button>
      ) : (
        <>
          <div className="hud-title" style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span>TRAFİK SİMÜLASYONU</span>
            <span style={{ cursor: 'pointer' }} onClick={() => { pause(); setOpen(false); }}>×</span>
          </div>
          {error && <div style={{ color: '#ff5577', fontSize: 12, marginBottom: 8 }}>{error}</div>}
          {report && (
            <div style={{ fontSize: 11, opacity: 0.8, marginBottom: 10, lineHeight: 1.5 }}>
              Graf: {report.graph.nodes} düğüm / {report.graph.edges} kenar · en büyük bağlı bileşen %{Math.round(report.graph.largestLengthShare * 100)}<br />
              Talep: {report.demand.buildings} bina, %{Math.round(report.demand.buildingCoverage * 100)} düğüme 75 m içinde · kat medyanı {report.demand.medianFloors}
            </div>
          )}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            <label><div style={LABEL}>Senaryo</div>
              <select style={FIELD} value={settings.scenario} onChange={(e) => configure({ scenario: e.target.value })}>
                <option value="S0">S0 · mevcut ağ</option>
                <option value="S1" disabled={!scenarios.S1}>S1 · yol kapatma</option>
                <option value="S2" disabled={!scenarios.S2}>S2 · yeni bağlantı</option>
              </select></label>
            <label><div style={LABEL}>Araç yoğunluğu</div>
              <select style={FIELD} value={settings.rate} onChange={(e) => configure({ rate: Number(e.target.value) })}>
                {Object.entries(DENSITIES).map(([k, v]) => <option key={k} value={v}>{{ low: 'Düşük', medium: 'Orta', high: 'Yüksek' }[k]} · {v} araç/s</option>)}
                <option value={1.5}>Tıkanıklık · 1,5 araç/s</option>
                <option value={2.5}>Doygun · 2,5 araç/s</option>
              </select></label>
            <label><div style={LABEL}>Tohum</div>
              <select style={FIELD} value={settings.seed} onChange={(e) => configure({ seed: Number(e.target.value) })}>
                {SEEDS.map((s) => <option key={s} value={s}>{s}</option>)}
              </select></label>
            <label><div style={LABEL}>Oynatma hızı</div>
              <select style={FIELD} value={settings.speed} onChange={(e) => configure({ speed: Number(e.target.value) })}>
                {[1, 2, 5, 10].map((s) => <option key={s} value={s}>{s}×</option>)}
              </select></label>
            <label style={{ gridColumn: '1 / 3' }}><div style={LABEL}>Kat sayısı varsayımı</div>
              <select style={FIELD} value={settings.floorsMode} onChange={(e) => configure({ floorsMode: e.target.value })}>
                <option value="median">OSM kat verisi, eksikte mahalle medyanı</option>
                <option value="one">Tüm binalarda kat = 1 (duyarlılık)</option>
              </select></label>
          </div>
          <div style={{ display: 'flex', gap: 8, margin: '12px 0' }}>
            <button className="cyber-button" style={{ flex: 1 }} disabled={!report || status === 'error'} onClick={running ? pause : play}>{running ? 'DURAKLAT' : 'BAŞLAT'}</button>
            <button className="cyber-button" disabled={!report} onClick={reset}>SIFIRLA</button>
          </div>
          {live && (
            <div style={{ fontSize: 13, lineHeight: 1.6 }}>
              Ağdaki araç: <b>{live.vehicles}</b> · ortalama hız: <b>{live.meanSpeedKmh.toFixed(1)} km/s</b> · girişte bekleyen: <b>{live.backlog}</b>
            </div>
          )}
          {settings.scenario === 'S1' && closure && (
            <div style={{ fontSize: 11, opacity: 0.75, marginTop: 6 }}>Kapatılan yol: {String(closure.wayId)} (kullanılan rotaların %{Math.round(closure.routeShare * 100)}'i{closure.disconnects ? '; ağı böler' : ''})</div>
          )}
          <div style={{ fontSize: 10.5, opacity: 0.6, marginTop: 10, lineHeight: 1.4 }}>
            Not: Sonuçlar senaryoları aynı varsayımlar altında birbiriyle karşılaştırmak içindir.
          </div>
        </>
      )}
    </div>
  );
}
