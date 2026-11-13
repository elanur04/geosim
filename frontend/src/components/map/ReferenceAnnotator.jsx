import React, { useEffect, useState } from 'react';
import { referenceBuildings, referenceCollection } from './referenceCoordinates';

const button = { width: '100%', padding: '9px', margin: '5px 0', background: '#102a30', color: '#e2e8f0', border: '1px solid #3b6068', cursor: 'pointer' };
export default function ReferenceAnnotator({ manifest, backendUrl, onClose }) {
  const [buildings, setBuildings] = useState([]), [points, setPoints] = useState([]);
  const [history, setHistory] = useState([]), [selected, setSelected] = useState(null);
  const [mode, setMode] = useState('select'), [complete, setComplete] = useState(false);
  const [revision, setRevision] = useState(0), [dirty, setDirty] = useState(false);
  const [busy, setBusy] = useState(true), [error, setError] = useState(''), [report, setReport] = useState(null), [zoom, setZoom] = useState(1);
  const [thresholdReport, setThresholdReport] = useState(null);
  const [review, setReview] = useState({ dataset_role: 'unassigned' });
  const [width, height] = manifest.image_size;
  useEffect(() => {
    const controller = new AbortController();
    (async () => {
      try {
        const response = await fetch(`${backendUrl}/api/references/${manifest.id}`, { signal: controller.signal });
        if (response.status === 404) return;
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Referans yüklenemedi.');
        setBuildings(referenceBuildings(data.collection, manifest)); setRevision(data.revision);
        setComplete(data.collection.meta.reference_complete === true);
        setReview(data.collection.meta);
      } catch (e) { if (e.name !== 'AbortError') setError(e.message); }
      finally { if (!controller.signal.aborted) setBusy(false); }
    })();
    return () => controller.abort();
  }, [manifest, backendUrl]);
  const change = next => {
    setHistory(h => [...h, { buildings, complete }]); setBuildings(next);
    setDirty(true); setComplete(false); setReport(null); setThresholdReport(null); setError('');
    setReview(r => ({ ...r, second_reviewer_fraction: 0, second_reviewer_area_fraction: 0, reviewed_at: null }));
  };
  const finish = () => {
    if (points.length < 3) { setError('Bir bina sınırı için en az üç köşe seçin.'); return; }
    if (mode === 'hole') change(buildings.map(b => b.id === selected ? { ...b, rings: [...b.rings, points] } : b));
    else {
      const id = `reference_${crypto.randomUUID()}`;
      change([...buildings, { id, rings: [points] }]); setSelected(id);
    }
    setMode('select'); setPoints([]);
  };
  const save = async () => {
    setBusy(true); setError('');
    try {
      const response = await fetch(`${backendUrl}/api/references/${manifest.id}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ revision, collection: referenceCollection(buildings, manifest, complete, review) }) });
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Referans kaydedilemedi.');
      setRevision(data.revision); setDirty(false); setHistory([]);
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  };
  const evaluate = async (compare = false) => {
    setBusy(true); setError('');
    try {
      const response = await fetch(`${backendUrl}/api/references/${manifest.id}/${compare ? 'compare-thresholds' : 'evaluate'}`, { method: 'POST' });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Ölçüm tamamlanamadı.');
      if (compare) { setThresholdReport(data); setReport(data.results[0]); }
      else { setReport(data); setThresholdReport(null); }
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  };
  const clickImage = event => {
    if (busy) return;
    if (mode === 'select') { setSelected(null); return; }
    const rect = event.currentTarget.getBoundingClientRect();
    setPoints(p => [...p, [(event.clientX - rect.left) / rect.width * width, (event.clientY - rect.top) / rect.height * height]]);
  };
  const exportReference = () => {
    const raw = JSON.stringify(referenceCollection(buildings, manifest, complete, review), null, 2);
    const url = URL.createObjectURL(new Blob([raw], { type: 'application/geo+json' })), a = document.createElement('a');
    a.href = url; a.download = `${manifest.id}-reference.geojson`; a.click(); URL.revokeObjectURL(url);
  };
  const exportReport = () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(thresholdReport ?? report, null, 2)], { type: 'application/json' })), a = document.createElement('a');
    a.href = url; a.download = `${manifest.id}-evaluation.json`; a.click(); URL.revokeObjectURL(url);
  };
  return <div role="dialog" aria-label="Bağımsız referans işaretleme" style={{ position: 'absolute', inset: 0, zIndex: 40, display: 'grid', gridTemplateColumns: 'minmax(0,1fr) 320px', background: '#071017', color: '#e2e8f0' }}>
    <div style={{ overflow: 'auto', padding: '12px' }}>
      <svg aria-label="Sabit uydu görüntüsü ve elle çizilmiş bina sınırları" viewBox={`0 0 ${width} ${height}`} onClick={clickImage}
        style={{ width: `${zoom * 100}%`, height: 'auto', aspectRatio: `${width}/${height}`, minWidth: '700px', display: 'block', cursor: mode === 'select' ? 'default' : 'crosshair' }}>
        <image href={`${backendUrl}/api/analysis-runs/${manifest.id}/image.png`} width={width} height={height} />
        {buildings.map(b => <path key={b.id} d={b.rings.map(r => `M ${r.map(p => p.join(',')).join(' L ')} Z`).join(' ')} fillRule="evenodd"
          fill={b.id === selected ? '#ffffff30' : '#00ff9d20'} stroke={b.id === selected ? '#fff' : '#00ff9d'} strokeWidth={3 / zoom}
          onClick={ev => { if (mode === 'select') { ev.stopPropagation(); setSelected(b.id); } }} />)}
        {points.length > 0 && <polyline points={points.map(p => p.join(',')).join(' ')} fill="none" stroke="#fbbf24" strokeWidth={4 / zoom} pointerEvents="none" />}
        {points.map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r={5 / zoom} fill="#fbbf24" pointerEvents="none" />)}
      </svg>
    </div>
    <aside style={{ overflowY: 'auto', padding: '18px', borderLeft: '1px solid #365058' }}>
      <h2 style={{ fontSize: '17px', color: '#00ff9d' }}>BAĞIMSIZ BİNA REFERANSI</h2>
      <p style={{ fontSize: '13px', lineHeight: 1.5 }}>Yalnızca arşivlenmiş görüntü gösteriliyor. OSM ve model önerileri gizli. Görüntünün tamamındaki her binayı tek tek çizin; avluları boşluk olarak işaretleyin.</p>
      <p>{buildings.length} bina · {dirty ? 'Kaydedilmedi' : `Kayıtlı sürüm ${revision}`}</p>
      {error && <p role="alert" style={{ color: '#fb7185', overflowWrap: 'anywhere' }}>{error}</p>}
      <fieldset disabled={busy} style={{ border: 0, padding: 0, margin: 0 }}>
        <button style={button} onClick={() => setZoom(z => Math.min(4, z + .5))}>YAKINLAŞTIR</button>
        <button style={button} onClick={() => setZoom(z => Math.max(.5, z - .5))}>UZAKLAŞTIR</button>
        <button style={button} onClick={() => { setMode('building'); setPoints([]); }}>YENİ BİNA ÇİZ</button>
        <button style={button} disabled={!selected} onClick={() => { setMode('hole'); setPoints([]); }}>SEÇİLİ BİNAYA AVLU ÇİZ</button>
        {mode !== 'select' && <>
          <p>{points.length} köşe seçildi. Sınırı sırayla tıklayın.</p>
          <button style={button} onClick={finish}>SINIRI TAMAMLA</button>
          <button style={button} onClick={() => setPoints(p => p.slice(0, -1))}>SON KÖŞEYİ GERİ AL</button>
          <button style={button} onClick={() => { setMode('select'); setPoints([]); }}>ÇİZİMİ İPTAL ET</button>
        </>}
        <button style={button} disabled={!selected || mode !== 'select'} onClick={() => { change(buildings.filter(b => b.id !== selected)); setSelected(null); }}>SEÇİLİ REFERANSI SİL</button>
        <button style={button} disabled={!history.length} onClick={() => {
          const previous = history.at(-1); setBuildings(previous.buildings); setComplete(previous.complete);
          setHistory(h => h.slice(0, -1)); setDirty(true); setReport(null); setSelected(null);
        }}>GERİ AL</button>
        <label style={{ display: 'block', margin: '12px 0' }}>Veri rolü
          <select aria-label="Referans veri rolü" value={review.dataset_role || 'unassigned'} onChange={e => { setReview(r => ({ ...r, dataset_role: e.target.value })); setDirty(true); setReport(null); setThresholdReport(null); }}>
            <option value="unassigned">Henüz atanmadı / ön deneme</option>
            <option value="train">Eğitim</option><option value="development">D1 geliştirme</option><option value="test">Nihai test</option>
          </select>
        </label>
        <p>Gerçekleşen ikinci kontrolü kaydedin. Çizimler değişirse kontrol oranları sıfırlanır.</p>
        {[['second_reviewer', 'Kontrol eden'], ['reviewed_at', 'Kontrol tarihi'], ['second_reviewer_fraction', 'Kontrol edilen bina oranı (0–1)'], ['second_reviewer_area_fraction', 'Kontrol edilen alan oranı (0–1)']].map(([key, title]) =>
          <label key={key} style={{ display: 'block', margin: '8px 0' }}>{title}
            <input aria-label={title} type={key.includes('fraction') ? 'number' : key === 'reviewed_at' ? 'date' : 'text'}
              min={key.includes('fraction') ? 0 : undefined} max={key.includes('fraction') ? 1 : undefined} step="0.01"
              value={review[key] ?? ''} onChange={e => { setReview(r => ({ ...r, [key]: e.target.value })); setDirty(true); setReport(null); setThresholdReport(null); }} />
          </label>)}
        <label style={{ display: 'block', margin: '15px 0', lineHeight: 1.5 }}><input type="checkbox" checked={complete} disabled={!buildings.length || mode !== 'select'}
          onChange={ev => { setComplete(ev.target.checked); setDirty(true); setReport(null); setThresholdReport(null); }} /> Görüntünün tamamındaki tüm binaları işaretledim.</label>
        <button style={button} disabled={mode !== 'select'} onClick={save}>REFERANSI KAYDET</button>
        <button style={button} disabled={dirty || !complete || !revision} onClick={() => evaluate()}>OSM / AI / HİBRİT F1 ÖLÇ</button>
        <button style={button} disabled={dirty || !complete || !revision || review.dataset_role !== 'development'} onClick={() => evaluate(true)}>D1 AI F1 İLE EŞİKLERİ KARŞILAŞTIR</button>
        <button style={button} disabled={mode !== 'select'} onClick={exportReference}>REFERANS GEOJSON İNDİR</button>
      </fieldset>
      {report && <div style={{ borderTop: '1px solid #365058', marginTop: '12px', paddingTop: '12px' }}>
        {thresholdReport && <div>{thresholdReport.results.map(r => <p key={r.analysis_run_id}>Eşik {r.inference_confidence}: AI F1 %{(r.methods.ai.f1 * 100).toFixed(1)}</p>)}<p>{thresholdReport.note}</p></div>}
        {Object.entries(report.methods).map(([method, r]) => <div key={method}>
          <p>{method.toUpperCase()} F1: %{(r.f1 * 100).toFixed(1)} · TP {r.tp}, FP {r.fp}, FN {r.fn}</p>
          <p style={{ color: '#8ba4b8', fontSize: '12px' }}>Hassasiyet: %{(r.precision * 100).toFixed(1)} · Duyarlılık: %{(r.recall * 100).toFixed(1)} · Eşleşen IoU: {r.mean_matched_iou?.toFixed(3) ?? 'Yok'}</p>
        </div>)}
        <p>Hibrit − OSM: {report.hybrid_minus_osm_f1_pp.toFixed(1)} puan</p>
        <p>%95 güven aralığı: {report.paired_ci95_pp?.map(v => v.toFixed(1)).join(' / ') ?? 'Hesaplanamadı'}</p>
        <p>Hibrit − AI: {report.hybrid_minus_ai_f1_pp?.toFixed(1)} puan</p>
        <p>%95 güven aralığı: {report.paired_ci95_hybrid_minus_ai_pp?.map(v => v.toFixed(1)).join(' / ') ?? 'Hesaplanamadı'}</p>
        {!report.advisor_review_requirement_met && <p style={{ color: '#fbbf24' }}>Formdaki %20 danışman kontrolü henüz kaydedilmedi.</p>}
        <button style={button} onClick={exportReport}>ÖLÇÜM RAPORUNU İNDİR</button>
      </div>}
      <p style={{ color: '#8ba4b8', fontSize: '12px' }}>Kaynak: Mapbox uydu görüntüsü. Çekim tarihi sağlayıcıdan bilinmiyor; arşivlenme tarihi: {new Date(manifest.captured_at).toLocaleString('tr-TR')}.</p>
      <button style={button} disabled={dirty || busy || mode !== 'select'} onClick={onClose}>HARİTAYA DÖN</button>
      {(dirty || mode !== 'select') && <button style={{ ...button, color: '#fb7185' }} disabled={busy} onClick={onClose}>KAYDEDİLMEMİŞ ÇİZİMDEN VAZGEÇ VE KAPAT</button>}
    </aside>
  </div>;
}
