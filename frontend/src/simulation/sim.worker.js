// Tüm fizik hesapları arayüz iş parçacığının dışında çalışır (araştırma önerisi: Web Worker).
import { createSimController } from './simController.js';

const controller = createSimController((message, transfer) => self.postMessage(message, transfer));
let last = performance.now();

self.onmessage = (event) => {
  try { controller.handle(event.data); }
  catch (error) { self.postMessage({ type: 'error', message: String(error?.message ?? error) }); }
};

// ~30 Hz; her çağrıda geçen gerçek süre kadar sabit 0,1 s fizik adımı atılır.
setInterval(() => {
  const now = performance.now(), elapsed = (now - last) / 1000; last = now;
  try { controller.tick(elapsed); }
  catch (error) { self.postMessage({ type: 'error', message: String(error?.message ?? error) }); }
}, 33);
