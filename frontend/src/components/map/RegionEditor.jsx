import React, { useEffect } from 'react';

const fieldStyle = { width: '100%', background: '#0b1820', border: '1px solid #365058', color: '#e2e8f0', padding: '7px', margin: '4px 0 10px' };
const buttonStyle = { width: '100%', background: 'rgba(0,255,157,0.08)', color: '#00ff9d', border: '1px solid #365058', padding: '8px', marginBottom: '7px', cursor: 'pointer', textAlign: 'left' };
const hintStyle = { color: '#8ba4b8', lineHeight: 1.5, marginBottom: '8px' };

const SOURCE_LABELS = { osm: 'OpenStreetMap', manual: 'Elle çizildi', roboflow: 'Yapay zekâ tespiti' };
const HEIGHT_LABELS = { osm: 'OSM', manual: 'Elle girildi' };

export default function RegionEditor({ editor, collection, tooltip }) {
  const e = editor, f = e.selected;
  const { loadRegions } = editor;
  const drawing = e.mode.startsWith('add_') || e.mode.startsWith('replace_');
  useEffect(() => { loadRegions(); }, [loadRegions]);
  return <fieldset disabled={e.busy} style={{ fontSize: '13px', border: 0, padding: 0, margin: 0, minWidth: 0 }}>
    <label>Kayıtlı çalışma alanları<select aria-label="Kayıtlı çalışma alanları" style={fieldStyle} disabled={e.dirty} value={e.region?.id ?? ''}
      onChange={ev => { if (ev.target.value) e.openRegion(ev.target.value); }}>
      <option value="">Kayıt seçin</option>{e.regions.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}
    </select></label>
    <label>Çalışma alanı adı<input aria-label="Çalışma alanı adı" style={fieldStyle} value={e.name} onChange={ev => e.setName(ev.target.value)} /></label>
    <div role="status" style={{ color: e.dirty ? '#fbbf24' : '#8ba4b8', marginBottom: '10px' }}>
      {e.busy ? 'Kaydediliyor…' : e.dirty ? 'Kaydedilmemiş değişiklikler var' : e.region ? `Kayıtlı sürüm: ${e.region.revision}` : 'Henüz kaydedilmedi'}
    </div>
    {e.region?.revision > 1 && <button style={buttonStyle} disabled={e.dirty} onClick={e.restorePrevious}>ÖNCEKİ SÜRÜME DÖN</button>}
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '5px' }}>
      <button style={buttonStyle} disabled={e.busy || !collection} onClick={e.save}>KAYDET</button>
      <button style={buttonStyle} disabled={!e.undoCount || e.busy} onClick={e.undo}>GERİ AL ({e.undoCount})</button>
      <button style={buttonStyle} disabled={!collection || e.busy} onClick={() => e.beginDrawing('building')}>BİNA ÇİZ</button>
      <button style={buttonStyle} disabled={!collection || e.busy} onClick={() => e.beginDrawing('road')}>YOL ÇİZ</button>
    </div>
    {drawing && <div style={{ border: '1px solid #fbbf24', padding: '8px', marginBottom: '12px' }}>
      Haritada köşelere sırayla tıklayın. {e.points.length} nokta seçildi.
      <button style={buttonStyle} onClick={e.finishDrawing}>ÇİZİMİ TAMAMLA</button>
      <button style={buttonStyle} onClick={e.removeLastPoint}>SON NOKTAYI GERİ AL</button>
      <button style={buttonStyle} onClick={e.cancelDrawing}>ÇİZİMİ İPTAL ET</button>
    </div>}
    {e.mode === 'move_vertex' && <div style={{ color: '#fbbf24', margin: '10px 0' }}>
      Seçilen köşenin yeni konumuna tıklayın.
      <button style={buttonStyle} onClick={e.cancelDrawing}>KÖŞE TAŞIMAYI İPTAL ET</button>
    </div>}
    {e.mode === 'move_feature' && <div style={{ color: '#fbbf24', margin: '10px 0' }}>
      Nesnenin merkezini taşımak istediğiniz konuma tıklayın.
      <button style={buttonStyle} onClick={e.cancelDrawing}>TAŞIMAYI İPTAL ET</button>
    </div>}
    <div style={{ borderTop: '1px solid #365058', paddingTop: '12px', marginTop: '5px' }}>
      {f ? <>
        <div style={{ color: '#00ff9d', overflowWrap: 'anywhere', marginBottom: '8px' }}>SEÇİLİ {f.properties.type === 'building' ? 'BİNA' : 'YOL'}</div>
        <div style={hintStyle}>Kaynak: {SOURCE_LABELS[f.properties.source] ?? 'Yapay zekâ tespiti'}</div>
        <label>Nesne adı<input aria-label="Nesne adı" style={fieldStyle} value={f.properties.name ?? ''} onChange={ev => e.patchSelected({ name: ev.target.value })} /></label>
        {f.properties.type === 'building' ? <>
          <label>Yükseklik (m)<input aria-label="Yükseklik" type="number" min="1" max="1000" style={fieldStyle} value={f.properties.height ?? ''}
            onChange={ev => e.patchSelected({ height: ev.target.value === '' ? null : Number(ev.target.value), height_source: 'manual' })} /></label>
          <div style={hintStyle}>Yükseklik kaynağı: {HEIGHT_LABELS[f.properties.height_source] ?? 'Tahmini'}</div>
        </> : <>
          <label>Yol genişliği (m)<input aria-label="Yol genişliği" type="number" min="0.5" max="100" style={fieldStyle} value={f.properties.width ?? 3}
            onChange={ev => e.patchSelected({ width: Number(ev.target.value) })} /></label>
          <label>Yön<select aria-label="Yol yönü" style={fieldStyle} value={f.properties.oneway ?? 'no'} onChange={ev => e.patchSelected({ oneway: ev.target.value })}>
            <option value="no">Çift yön</option><option value="yes">Çizim yönünde tek yön</option><option value="-1">Çizimin tersine tek yön</option>
          </select></label>
        </>}
        <div style={hintStyle}>Sınırı düzeltmek için haritadaki beyaz köşeye, ardından yeni konumuna tıklayın.</div>
        <button style={buttonStyle} disabled={drawing} onClick={() => e.beginDrawing(f.properties.type, true)}>SINIRI YENİDEN ÇİZ</button>
        <button style={buttonStyle} disabled={e.mode !== 'select'} onClick={e.beginMove}>NESNEYİ TAŞI</button>
        <button style={{ ...buttonStyle, color: '#fb7185' }} disabled={e.busy} onClick={e.removeSelected}>NESNEYİ SİL</button>
      </> : <p style={hintStyle}>Düzenlemek için haritadaki bir bina veya yola tıklayın.{tooltip ? <><br /><span style={{ color: '#e2e8f0' }}>{tooltip.text}</span></> : null}</p>}
    </div>
    <button style={{ ...buttonStyle, marginTop: '12px' }} disabled={!collection} onClick={e.exportData}>GEOJSON OLARAK İNDİR</button>
  </fieldset>;
}
