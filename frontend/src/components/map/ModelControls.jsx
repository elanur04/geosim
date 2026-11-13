import React, { useEffect, useState } from 'react';

export default function ModelControls({ backendUrl, provider, setProvider, confidence, setConfidence, disabled, meta }) {
  const [providers, setProviders] = useState([]);
  const [error, setError] = useState('');
  useEffect(() => {
    const controller = new AbortController();
    fetch(`${backendUrl}/model-status`, { signal: controller.signal }).then(async response => {
      if (!response.ok) throw new Error('Model durumu alınamadı.');
      setProviders((await response.json()).providers || []);
    }).catch(e => { if (e.name !== 'AbortError') setError(e.message); });
    return () => controller.abort();
  }, [backendUrl]);
  const names = { roboflow: 'Roboflow · mevcut model', geoai: 'GeoAI · yerel bina modeli', yolo: 'YOLO · yerel eğitim' };
  return <fieldset disabled={disabled} style={{ border: '1px solid #365058', padding: 10, marginBottom: 12 }}>
    <legend>Bina modeli</legend>
    <select aria-label="Bina modeli" value={provider} onChange={e => setProvider(e.target.value)} style={{ width: '100%', background: '#102a30', color: '#e2e8f0', padding: 8 }}>
      {(providers.length ? providers : [{ provider: 'roboflow', ready: true }]).map(p =>
        <option key={p.provider} value={p.provider} disabled={!p.ready}>{names[p.provider]}{p.ready ? '' : ' · yapılandırılmadı'}</option>)}
    </select>
    <label style={{ display: 'block', marginTop: 8 }}>Güven eşiği
      <select aria-label="Model güven eşiği" value={confidence} onChange={e => setConfidence(Number(e.target.value))} style={{ marginLeft: 8 }}>
        <option value={0.25}>0.25</option><option value={0.4}>0.40</option><option value={0.6}>0.60</option>
      </select>
    </label>
    <p style={{ fontSize: 12, color: '#8ba4b8' }}>Bina sınırları korunur; renk kuralları yeni bina eklemez. Model değişikliği sonraki analizde uygulanır. Yerel doğruluk henüz ölçülmedi.</p>
    {meta?.model_provider && <p style={{ fontSize: 12 }}>Görüntülenen sonuç: {names[meta.model_provider]} · {meta.ai_processed_detections ?? 0} AI bina</p>}
    {error && <p role="alert">{error}</p>}
  </fieldset>;
}
