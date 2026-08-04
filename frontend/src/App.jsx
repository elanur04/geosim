import React, { useState, useEffect, useRef } from 'react';
import { Sun, Moon, Crosshair, Video, Layers, Settings, Truck, Upload, Sparkles, Wifi, Loader2 } from 'lucide-react';

function ThemeToggle() {
  const [isDark, setIsDark] = useState(() => {
    if (typeof window !== 'undefined') {
      return document.documentElement.classList.contains('dark');
    }
    return false;
  });

  useEffect(() => {
    const root = window.document.documentElement;
    if (isDark) {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
  }, [isDark]);

  return (
    <button
      onClick={() => setIsDark(!isDark)}
      className="p-2 rounded-full bg-white/10 dark:bg-black/20 hover:bg-white/20 dark:hover:bg-black/40 backdrop-blur-md transition-colors duration-200 text-green-500 focus:outline-none border border-white/20 dark:border-white/10"
      aria-label="Toggle Theme"
    >
      {isDark ? <Sun size={20} /> : <Moon size={20} />}
    </button>
  );
}

function Sidebar() {
  const navItems = [
    { icon: Crosshair, label: 'İHA Analizi', active: true },
    { icon: Video, label: 'Simülasyon' },
    { icon: Layers, label: 'Veri Görünümü' },
    { icon: Settings, label: 'Ayarlar' },
  ];

  return (
    <aside className="w-24 h-full glass rounded-2xl m-4 flex flex-col items-center py-6 hidden md:flex border-r border-white/10 dark:border-gray-800">
      <div className="mb-8 flex flex-col items-center">
        <div className="flex items-center text-green-500">
          <span className="text-3xl font-bold tracking-tighter">GS</span>
          <Wifi size={18} className="ml-1 animate-pulse" />
        </div>
      </div>
      
      <nav className="flex-1 w-full space-y-6 mt-4 flex flex-col items-center">
        {navItems.map((item, index) => (
          <a
            key={index}
            href="#"
            className={`flex flex-col items-center justify-center w-full group relative ${
              item.active ? 'text-green-500' : 'text-gray-500 dark:text-gray-400'
            }`}
          >
            {item.active && (
              <div className="absolute left-0 w-1 h-full bg-green-500 rounded-r-md shadow-[0_0_10px_rgba(34,197,94,0.5)]"></div>
            )}
            <div className={`p-3 rounded-xl transition-all duration-300 ${
              item.active 
                ? 'bg-green-500/10 shadow-[inset_0_0_10px_rgba(34,197,94,0.2)]' 
                : 'hover:bg-white/20 dark:hover:bg-white/5'
            }`}>
              <item.icon size={24} className={`transition-colors ${
                item.active ? 'text-green-500' : 'group-hover:text-green-400'
              }`} />
            </div>
            <span className={`text-[10px] font-medium mt-2 transition-colors ${
              item.active ? 'text-green-500' : 'group-hover:text-green-400'
            }`}>
              {item.label}
            </span>
          </a>
        ))}
      </nav>
    </aside>
  );
}

function App() {
  const fileInputRef = useRef(null);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [segmentationResult, setSegmentationResult] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    if (file) {
      // 1. Orijinal görüntüyü yerel olarak göster
      const localImageUrl = URL.createObjectURL(file);
      setUploadedImage(localImageUrl);
      setSegmentationResult(null);
      setErrorMsg("");
      setIsProcessing(true);

      // 2. Modeli çağırmak için backend API'sine istek gönder
      const formData = new FormData();
      formData.append("file", file);

      try {
        const response = await fetch("http://localhost:8000/predict", {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error("Model analizi başarısız oldu. API hatası.");
        }

        // 3. API'den dönen PNG'yi blob olarak al ve göster
        const blob = await response.blob();
        const resultImageUrl = URL.createObjectURL(blob);
        setSegmentationResult(resultImageUrl);
      } catch (err) {
        console.error("API Error:", err);
        setErrorMsg("API'ye bağlanırken hata oluştu. Backend'in çalıştığından emin olun.");
      } finally {
        setIsProcessing(false);
      }
    }
  };

  return (
    <div 
      className="min-h-screen bg-cover bg-center bg-no-repeat bg-fixed transition-colors duration-300 flex overflow-hidden font-sans text-gray-900 dark:text-gray-100 relative"
      style={{ backgroundImage: "url('/back2.jpg')" }}
    >
      {/* Overlay to keep dark/light mode styles intact and readable */}
      <div className="absolute inset-0 bg-gray-50/75 dark:bg-slate-950/75 backdrop-blur-[1px] pointer-events-none z-0"></div>

      <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none z-0">
        <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] rounded-full bg-green-500/10 blur-[150px]"></div>
        <div className="absolute bottom-[-20%] right-[-10%] w-[50%] h-[50%] rounded-full bg-green-500/20 blur-[150px]"></div>
      </div>

      <div className="relative z-10 flex w-full h-screen p-2 md:p-4">
        <Sidebar />
        
        <main className="flex-1 h-full flex flex-col overflow-y-auto px-2 md:px-6 py-4">
          <header className="relative flex justify-center items-center mb-8 h-12">
            <h2 className="text-xl md:text-2xl font-bold tracking-tight text-gray-800 dark:text-white text-center px-12">
              İHA Görüntüleri ile Yapay Zekâ Tabanlı Kentsel Analiz ve Simülasyon
            </h2>
            <div className="absolute right-0 flex items-center">
              <ThemeToggle />
            </div>
          </header>
          
          <div className="flex-1 w-full max-w-7xl mx-auto space-y-6">
            
            {/* Upload Section */}
            <div className="glass rounded-xl p-3 md:p-4 border border-dashed border-green-500/50 bg-green-50/30 dark:bg-green-900/10 flex flex-row justify-between items-center transition-all duration-300 hover:shadow-[0_0_20px_rgba(34,197,94,0.3)] hover:border-green-500">
              <div className="flex items-center">
                <h3 className="text-base md:text-lg font-semibold dark:text-gray-200 px-2">İHA Görüntüsü Yükle</h3>
                {errorMsg && <p className="text-sm text-red-500 ml-4 font-medium">{errorMsg}</p>}
              </div>
              
              <input 
                type="file" 
                accept="image/*" 
                ref={fileInputRef} 
                onChange={handleImageUpload} 
                className="hidden" 
              />
              <button 
                onClick={() => fileInputRef.current?.click()}
                disabled={isProcessing}
                className={`px-4 py-1.5 md:px-5 md:py-2 rounded-lg ${isProcessing ? 'bg-gray-400/50 cursor-not-allowed' : 'bg-white/50 dark:bg-white/10 hover:bg-white/80 dark:hover:bg-white/20 border-green-500/50 hover:border-green-500 hover:shadow-[0_0_15px_rgba(34,197,94,0.4)] active:scale-105 active:shadow-[0_0_25px_rgba(34,197,94,0.6)] cursor-pointer transform hover:scale-105'} border transition-all duration-300 text-sm font-medium flex items-center space-x-2`}
              >
                {isProcessing ? <Loader2 size={16} className="animate-spin" /> : <Upload size={16} />}
                <span>{isProcessing ? 'İşleniyor...' : 'Görsel Yükle'}</span>
              </button>
            </div>
            
            {/* Main Panels */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              
              {/* Orijinal Görüntü */}
              <div className="glass rounded-2xl overflow-hidden flex flex-col border border-green-500/40 transition-all duration-300 hover:shadow-[0_0_20px_rgba(34,197,94,0.4)] hover:border-green-500 group cursor-pointer">
                <div className="p-4 border-b border-gray-200/50 dark:border-gray-800/50 bg-white/30 dark:bg-black/20">
                  <h3 className="text-sm font-semibold tracking-wide text-gray-700 dark:text-gray-300">
                    ORİJİNAL İHA GÖRÜNTÜSÜ
                  </h3>
                </div>
                <div className="relative h-64 md:h-80 w-full bg-transparent overflow-hidden flex items-center justify-center">
                  {uploadedImage ? (
                    <div 
                      className="absolute inset-0 bg-contain bg-no-repeat bg-center opacity-90 transition-transform duration-700"
                      style={{ backgroundImage: `url(${uploadedImage})` }}
                    ></div>
                  ) : (
                    <div className="text-gray-400 flex flex-col items-center">
                      <Upload size={32} className="mb-2 opacity-50" />
                      <p className="text-sm">Analiz edilecek görüntüyü yükleyin</p>
                    </div>
                  )}
                </div>
              </div>

              {/* Segmentasyon Haritası */}
              <div className="glass rounded-2xl overflow-hidden flex flex-col border border-green-500/40 transition-all duration-300 hover:shadow-[0_0_20px_rgba(34,197,94,0.4)] hover:border-green-500 group cursor-pointer">
                <div className="p-4 border-b border-gray-200/50 dark:border-gray-800/50 bg-white/30 dark:bg-black/20">
                  <h3 className="text-sm font-semibold tracking-wide text-gray-700 dark:text-gray-300 flex items-center">
                    GEO-AI ÇOK SINIFLI SEGMENTASYON HARİTASI
                    {isProcessing && <Loader2 size={14} className="ml-2 animate-spin text-green-500" />}
                  </h3>
                </div>
                <div className="relative h-64 md:h-80 w-full bg-transparent overflow-hidden flex items-center justify-center">
                  
                  {isProcessing ? (
                    <div className="flex flex-col items-center justify-center space-y-4">
                      <div className="relative w-16 h-16">
                        <div className="absolute inset-0 border-4 border-t-green-500 border-r-transparent border-b-green-500 border-l-transparent rounded-full animate-spin"></div>
                        <div className="absolute inset-2 border-4 border-t-transparent border-r-fuchsia-500 border-b-transparent border-l-fuchsia-500 rounded-full animate-spin-reverse"></div>
                      </div>
                      <p className="text-green-500 font-medium animate-pulse tracking-widest text-sm">YAPAY ZEKA MODELİ ÇALIŞIYOR...</p>
                    </div>
                  ) : segmentationResult ? (
                    <div 
                      className="absolute inset-0 bg-contain bg-no-repeat bg-center"
                      style={{ backgroundImage: `url(${segmentationResult})` }}
                    ></div>
                  ) : (
                    <div className="absolute inset-0 flex items-center justify-center opacity-10">
                      <Layers size={80} className="text-green-500" />
                    </div>
                  )}
                  {/* Legend Overlay removed from inside the map */}
                </div>
                
                {/* External Legend Panel */}
                {(segmentationResult || isProcessing) && (
                  <div className="border-t border-gray-200/50 dark:border-gray-800/50 bg-white/30 dark:bg-black/20 px-5 py-3 flex flex-wrap justify-between items-center text-xs font-semibold tracking-wide text-gray-700 dark:text-gray-300 w-full mt-auto">
                    <div className="flex items-center"><div className="w-3.5 h-3.5 rounded bg-[rgb(128,0,0)] mr-2 shadow-[0_0_5px_rgba(128,0,0,0.5)] border border-black/10 dark:border-white/10"></div>Bina</div>
                    <div className="flex items-center"><div className="w-3.5 h-3.5 rounded bg-[rgb(0,128,0)] mr-2 shadow-[0_0_5px_rgba(0,128,0,0.5)] border border-black/10 dark:border-white/10"></div>Ağaç</div>
                    <div className="flex items-center"><div className="w-3.5 h-3.5 rounded bg-[rgb(128,128,0)] mr-2 shadow-[0_0_5px_rgba(128,128,0,0.5)] border border-black/10 dark:border-white/10"></div>Yol</div>
                    <div className="flex items-center"><div className="w-3.5 h-3.5 rounded bg-[rgb(192,0,192)] mr-2 shadow-[0_0_5px_rgba(192,0,192,0.5)] border border-black/10 dark:border-white/10"></div>Araç</div>
                  </div>
                )}
              </div>

            </div>
            
            {/* Bottom Section */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              
              {/* Kentsel Metrikler */}
              <div className="glass rounded-2xl p-5 border border-green-500/40 relative overflow-hidden group transition-all duration-300 hover:scale-[1.03] hover:border-green-500 hover:shadow-[0_0_20px_rgba(34,197,94,0.5)] active:scale-105 active:shadow-[0_0_30px_rgba(34,197,94,0.8)] cursor-pointer">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="font-semibold text-gray-700 dark:text-gray-300">Kentsel Metrikler</h3>
                  <button className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200">...</button>
                </div>
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Bina Alanı</p>
                    <p className="text-lg font-bold text-gray-800 dark:text-white">{segmentationResult ? "23.1 m²" : "-"}</p>
                  </div>
                  <div className="border-l border-gray-200 dark:border-gray-800 pl-4">
                    <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Yol Uzunluğu</p>
                    <p className="text-lg font-bold text-gray-800 dark:text-white">{segmentationResult ? "471 m" : "-"}</p>
                  </div>
                  <div className="border-l border-gray-200 dark:border-gray-800 pl-4">
                    <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Ağaç Sayısı</p>
                    <p className="text-lg font-bold text-gray-800 dark:text-white">{segmentationResult ? "30 Adet" : "-"}</p>
                  </div>
                </div>
              </div>

              {/* Araç Tespiti */}
              <div className="glass rounded-2xl p-5 border border-green-500/40 relative overflow-hidden group transition-all duration-300 hover:scale-[1.03] hover:border-green-500 hover:shadow-[0_0_20px_rgba(34,197,94,0.5)] active:scale-105 active:shadow-[0_0_30px_rgba(34,197,94,0.8)] cursor-pointer">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="font-semibold text-gray-700 dark:text-gray-300">Araç Tespiti</h3>
                  <button className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200">...</button>
                </div>
                <div className="flex items-center space-x-4 mt-2">
                  <div className="p-3 bg-green-500/10 rounded-xl border border-green-500/20 text-green-500 shadow-[inset_0_0_15px_rgba(34,197,94,0.1)] group-hover:scale-110 transition-transform">
                    <Truck size={24} />
                  </div>
                  <p className="text-2xl font-bold text-gray-800 dark:text-white tracking-tight">{segmentationResult ? "1.602 Adet" : "-"}</p>
                </div>
              </div>

              {/* Kentsel Simülasyon Paneli */}
              <div className="glass rounded-2xl p-5 border border-green-500/40 relative overflow-hidden transition-all duration-300 hover:scale-[1.03] hover:border-green-500 hover:shadow-[0_0_20px_rgba(34,197,94,0.5)] active:scale-105 active:shadow-[0_0_30px_rgba(34,197,94,0.8)] cursor-pointer">
                <div className="flex justify-between items-center mb-4 relative z-10">
                  <h3 className="font-semibold text-gray-700 dark:text-gray-300">Kentsel Simülasyon Paneli</h3>
                  <button className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200">...</button>
                </div>
                <div className="flex space-x-3 mt-4 relative z-10">
                  <button className="flex-1 py-2 px-4 rounded-xl bg-gradient-to-r from-green-600 to-green-500 hover:from-green-500 hover:to-green-400 text-white font-medium shadow-[0_0_15px_rgba(34,197,94,0.4)] transition-all duration-300 transform hover:scale-105 active:scale-110 active:shadow-[0_0_25px_rgba(34,197,94,0.8)] cursor-pointer">
                    Run Simulation
                  </button>
                  <button className="flex-1 py-2 px-4 rounded-xl bg-white/10 hover:bg-white/20 dark:bg-black/30 dark:hover:bg-black/50 border border-green-500/40 hover:border-green-500 hover:shadow-[0_0_15px_rgba(34,197,94,0.4)] font-medium transition-all duration-300 transform hover:scale-105 active:scale-110 active:shadow-[0_0_25px_rgba(34,197,94,0.7)] text-gray-700 dark:text-gray-200 cursor-pointer">
                    View Results
                  </button>
                </div>
              </div>

            </div>

            <div className="mt-6 text-xs text-gray-500 dark:text-gray-500">
              Model: ResNet34 U-Net | Sınıflar: 8 | Durum: {isProcessing ? 'İşleniyor...' : 'Hazır'}
            </div>

          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
