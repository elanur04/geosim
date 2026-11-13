// React köprüsü: yol/bina verisini Worker'a verir, araç karelerini ve özetleri döndürür.
import { useCallback, useEffect, useRef, useState } from 'react';

export default function useTrafficSimulation({ roads, buildings, enabled }) {
  const workerRef = useRef(null);
  const [state, setState] = useState({ status: 'idle', report: null, frame: null, live: null, error: '' });
  const [settings, setSettings] = useState({ scenario: 'S0', rate: 0.35, seed: 202601, speed: 1, floorsMode: 'median' });

  const post = useCallback((message) => workerRef.current?.postMessage(message), []);

  useEffect(() => {
    if (!enabled) return undefined;
    const worker = new Worker(new URL('./sim.worker.js', import.meta.url), { type: 'module' });
    workerRef.current = worker;
    worker.onmessage = ({ data }) => {
      if (data.type === 'ready') setState((s) => ({ ...s, status: 'ready', report: data, error: '' }));
      else if (data.type === 'frame') setState((s) => ({ ...s, frame: data, live: data.live }));
      else if (data.type === 'error') setState((s) => ({ ...s, status: 'error', error: data.message }));
    };
    return () => { worker.terminate(); workerRef.current = null; };
  }, [enabled]);

  // Veri ya da talep varsayımı değişince ağı yeniden kur
  useEffect(() => {
    if (!enabled || !workerRef.current || !roads?.length) return;
    setState((s) => ({ ...s, status: 'building', error: '' }));
    post({ type: 'init', roads, buildings, floorsMode: settings.floorsMode, rate: settings.rate, seed: settings.seed, scenario: settings.scenario });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, roads, buildings, settings.floorsMode, post]);

  const configure = useCallback((patch) => { setSettings((s) => ({ ...s, ...patch })); if (!('floorsMode' in patch)) post({ type: 'configure', ...patch }); }, [post]);
  const play = useCallback(() => { post({ type: 'play' }); setState((s) => ({ ...s, status: 'playing' })); }, [post]);
  const pause = useCallback(() => { post({ type: 'pause' }); setState((s) => ({ ...s, status: 'paused' })); }, [post]);
  const reset = useCallback(() => { post({ type: 'reset' }); setState((s) => ({ ...s, status: 'ready' })); }, [post]);

  return { ...state, settings, configure, play, pause, reset };
}
