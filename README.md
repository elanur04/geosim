# GeoSim — Kentsel Dijital İkiz ve Trafik Simülasyonu

GeoSim, seçilen bir bölgenin bina ve yol ağını çıkarıp harita üzerinde düzenlenebilir
bir dijital ikiz oluşturan ve bu ağ üzerinde mikroskobik trafik simülasyonu çalıştıran
bir web uygulamasıdır.

- **Bina tespiti:** OpenStreetMap (Overpass) bina poligonları temel alınır; OSM'de eksik
  olan binalar Mapbox uydu görüntüsü üzerinde yapay zekâ ile (Roboflow instance
  segmentation) tespit edilerek eklenir.
- **Yol ağı:** OSM yolları, kavşak bağlantılarıyla birlikte yönlü bir grafa dönüştürülür.
- **Düzenleme:** Bina/yol çizme, taşıma, silme, yükseklik ve yön bilgisi girme; çalışma
  alanını sürümlü olarak kaydetme ve GeoJSON olarak indirme.
- **Trafik simülasyonu:** Akıllı Sürücü Modeli (IDM), Dijkstra rotalama ve bina alanı ×
  kat sayısına göre talep üretimi. Üç senaryo karşılaştırılır: S0 mevcut ağ,
  S1 yol kapatma, S2 yeni bağlantı yolu. Simülasyon tarayıcıda Web Worker içinde çalışır.

## Gereksinimler

- **Python 3.10+** — <https://www.python.org/downloads/> (kurulumda *Add Python to PATH* seçilmeli)
- **Node.js 18+** — <https://nodejs.org/> (LTS sürümü)
- İnternet bağlantısı (uydu görüntüsü, OSM ve yapay zekâ servisi için)

API anahtarları `backend/.env` ve `frontend/.env` dosyalarında hazır olarak bulunur.

## Çalıştırma

İki ayrı terminal gerekir; uygulama çalıştığı sürece ikisi de açık kalmalıdır.

**1. terminal — backend**

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r backend/requirements.txt
cd backend
uvicorn api:app --port 8000
```

**2. terminal — frontend**

```bash
cd frontend
npm install
npm run dev
```

Uygulama tarayıcıda <http://localhost:5173> adresinde açılır. Paket kurulumu (`pip install`,
`npm install`) yalnızca ilk çalıştırmada gereklidir.

## Kullanım

1. Açılışta kayıtlı **Bartın Merkez** çalışma alanı yüklenir.
2. Sol panelden yer arayabilir, koordinat girebilir veya hazır bölgelerden birini
   seçebilirsiniz. Seçilen bölgede önce OSM binaları ve yollar hemen çizilir, ardından
   uydu görüntüsü yapay zekâ ile taranır ve eksik binalar eklenir (~15–30 sn).
3. Haritada binalar kaynağa göre renklendirilir: mavi OSM, yeşil yapay zekâ ile
   eklenen, sarı elle çizilen binalar; turuncu çizgiler yol ağıdır.
4. Sağ panelden bina/yol seçip düzenleyebilir, yeni nesne çizebilir ve çalışma alanını
   kaydedebilirsiniz.
5. **Trafik Simülasyonu** düğmesiyle senaryo, araç yoğunluğu ve oynatma hızını seçip
   simülasyonu başlatabilirsiniz.

## Proje yapısı

```
backend/                    FastAPI sunucusu
  api.py                    Uç noktalar: /analyze, /api/roads, /api/search, /model-status
  vectorize.py              Uydu görüntüsü taraması, maske → poligon, OSM + YZ hibrit birleştirme
  osm_source.py             Overpass'tan özgün OSM bina ve yol geometrileri
  prediction_pipeline.py    Tespitlerin son işlenmesi (yinelenen maskeleri ayıklama)
  building_models.py        Model sağlayıcıları (Roboflow; isteğe bağlı yerel modeller)
  regions.py                Kayıtlı çalışma alanları (SQLite, sürümlü)
  analysis_archive.py       Analiz girdilerinin arşivlenmesi
  evaluate_buildings.py     Elle çizilmiş referansa göre doğruluk (F1) değerlendirmesi
  verification/             Bartın ve İstanbul deney çıktıları
  test_*.py                 Birim testleri
frontend/                   React + Vite + deck.gl arayüzü
  src/components/map/       Harita, düzenleme ve simülasyon panelleri
  src/simulation/           Trafik simülasyonu (graf, IDM, rotalama, talep, senaryolar)
  tests/                    Simülasyon ve düzenleme testleri
  scripts/                  Komut satırından toplu simülasyon deneyleri
```

## Testler

```bash
cd backend && python -m unittest discover -p "test_*.py"
cd frontend && npm test
```

## İsteğe bağlı: yerel modeller

Varsayılan bina tespiti Roboflow üzerinden yapılır. Yerel GeoAI / YOLOv8 modelleri ve
model eğitimi için ek paketler gerekir:

```bash
pip install -r backend/requirements-ml.txt
```
