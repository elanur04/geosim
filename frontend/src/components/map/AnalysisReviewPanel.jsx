import React, { useCallback, useEffect, useState } from 'react';

const label = run => {
  const lat = run.request.latitude;
  const place = Math.abs(lat - 41.5518) < .0001 ? 'Kutlubey' : Math.abs(lat - 41.6374) < .0001 ? 'Bartın merkez'
    : Math.abs(lat - 41.0295) < .0001 ? 'İstanbul / Balat' : `${lat.toFixed(4)}, ${run.request.longitude.toFixed(4)}`;
  return `${place} · eşik ${run.request.confidence} · ${new Date(run.captured_at).toLocaleTimeString('tr-TR')} ${run.status === 'complete' ? '' : '(eksik girdi)'}`;
};

export default function AnalysisReviewPanel({ backendUrl, disabled, onOpen, onError }) {
  const [runs, setRuns] = useState([]), [selected, setSelected] = useState(''), [busy, setBusy] = useState(false);
  const refresh = useCallback(async () => {
    try {
      const response = await fetch(`${backendUrl}/api/analysis-runs`);
      if (!response.ok) throw new Error('Analiz arşivi listelenemedi.');
      setRuns((await response.json()).runs);
    } catch (error) { onError(error.message); }
  }, [backendUrl, onError]);
  useEffect(() => { refresh(); }, [refresh]);
  const open = async () => {
    setBusy(true);
    try {
      const response = await fetch(`${backendUrl}/api/analysis-runs/${selected}`);
      const manifest = await response.json();
      if (!response.ok) throw new Error(manifest.detail || 'Analiz açılamadı.');
      onOpen(manifest);
    } catch (error) { onError(error.message); }
    finally { setBusy(false); }
  };
  return <div style={{ borderTop: '1px solid #365058', paddingTop: '12px', marginTop: '15px' }}>
    <strong>BAĞIMSIZ DOĞRULUK ÖLÇÜMÜ</strong>
    <p style={{ color: '#8ba4b8', lineHeight: 1.5 }}>Referans, aynı kayıtlı görüntüde OSM/AI önerileri gösterilmeden çizilir. F1 ölçümü ancak tüm görüntü işaretlendikten sonra yapılır.</p>
    <select aria-label="Doğrulanacak analiz" value={selected} onChange={e => setSelected(e.target.value)} style={{ width: '100%', padding: '7px', background: '#0b1820', color: '#eee' }}>
      <option value="">Analiz seçin</option>{runs.map(r => <option key={r.id} value={r.id} disabled={r.status !== 'complete'}>{label(r)}</option>)}
    </select>
    <button className="cyber-button" style={{ width: '100%', marginTop: '8px' }} disabled={!selected || disabled || busy} onClick={open}>REFERANS ÇİZ / ÖLÇ</button>
    <button style={{ background: 'none', color: '#8ba4b8', border: 0, padding: '8px', cursor: 'pointer' }} onClick={refresh}>Arşiv listesini yenile</button>
    {disabled && <p style={{ color: '#fbbf24' }}>Referansa geçmeden önce harita düzenlemelerini kaydedin.</p>}
  </div>;
}
