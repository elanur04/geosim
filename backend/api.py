import io
import os
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
from contextlib import asynccontextmanager

# Global değişkenler
device = None
model = None

INDEX_TO_COLOR = {
    0: (0, 0, 0),
    1: (128, 0, 0),
    2: (0, 128, 0),
    3: (128, 128, 0),
    4: (64, 0, 128),
    5: (192, 0, 192),
    6: (64, 64, 0),
    7: (128, 64, 128),
}
NUM_CLASSES = len(INDEX_TO_COLOR)


# 3. Sunucu açıldığı an (startup event) çalışacak yaşam döngüsü
@asynccontextmanager
async def lifespan(app: FastAPI):
    global device, model
    print("FastAPI başlatılıyor, model belleğe alınıyor...")
    
    # Cihazı belirleme
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")
    
    # Modeli oluşturma (ResNet34 Unet)
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None, 
        in_channels=3,
        classes=NUM_CLASSES
    ).to(device)
    
    # Model ağırlıklarının yolu (api.py ile aynı dizinde olan best_model.pth)
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.pth")
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model ağırlıkları başarıyla yüklendi: {model_path}")
    else:
        print(f"UYARI: Model ağırlıkları bulunamadı: {model_path}. Lütfen önce train.py ile modeli eğitin.")
        
    model.eval() # Modeli tahmin moduna al
    
    yield # Uygulama çalışırken burada bekler
    
    # Kapatılma durumu
    print("FastAPI kapatılıyor, bellek temizleniyor...")
    model = None

# 1. FastAPI uygulamasını başlat
app = FastAPI(title="GeoSim Segmentation API", lifespan=lifespan)

# 2. CORS ayarlarını ekle (Frontend'in iletişim kurabilmesi için)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. /predict POST endpoint'i
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model henüz yüklenmedi.")
        
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Lütfen geçerli bir görüntü dosyası yükleyin.")
        
    try:
        # Gelen dosyayı okuma ve PIL Image'a çevirme
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Görüntü okunamadı: {str(e)}")
        
    # 5. Görüntüyü ön işleme (256x256 ve tensöre çevirme)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    try:
        # Modeli kullanarak tahmin yapma
        with torch.no_grad():
            output = model(input_tensor)
            # En yüksek olasılığa sahip sınıfı seç (0 veya 1)
            predicted_classes = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            
        # 6. Tahmin maskesini tekrar görselleştirip RGB'ye dönüştürme
        prediction_visual = np.zeros((predicted_classes.shape[0], predicted_classes.shape[1], 3), dtype=np.uint8)
        for idx, color in INDEX_TO_COLOR.items():
            prediction_visual[predicted_classes == idx] = color
            
        result_image = Image.fromarray(prediction_visual)
        
        # Sonucu byte formatında belleğe kaydetme (PNG olarak)
        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Frontend'e direkt olarak PNG görüntüsünü (binary) döndür
        return Response(content=img_byte_arr, media_type="image/png")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tahmin işlemi sırasında hata oluştu: {str(e)}")
