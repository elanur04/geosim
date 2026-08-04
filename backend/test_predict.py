import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import argparse
import os
import segmentation_models_pytorch as smp

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
def main():
    # 3. argparse kullanarak argüman alımı
    parser = argparse.ArgumentParser(description="Model ile tek bir görsel üzerinde tahmin (inference) yapar.")
    parser.add_argument("--image_path", type=str, 
                        default="archive/train/train/seq1/Images/000000.png", 
                        help="Tahmin yapılacak görselin yolu")
    args = parser.parse_args()

    # 4. Cihaz (CUDA veya CPU) belirleme
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")

    # 5. Modeli oluşturma ve ağırlıkları yükleme
    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=NUM_CLASSES).to(device)
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.pth")
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model ağırlıkları (best_model.pth) başarıyla yüklendi.")
    else:
        print(f"Uyarı: {model_path} bulunamadı. Rastgele ağırlıklarla tahmin yapılıyor.")
    
    model.eval() # Modeli değerlendirme (evaluation) moduna alma

    # 6. Görüntüyü açma ve ön işleme
    if not os.path.exists(args.image_path):
        print(f"Hata: Belirtilen görüntü bulunamadı: {args.image_path}")
        return

    try:
        image = Image.open(args.image_path).convert("RGB")
    except Exception as e:
        print(f"Görsel açılırken hata oluştu: {e}")
        return

    # train.py'deki gibi 256x256 boyutuna getirme ve tensöre çevirme
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # Tensöre çevir ve unsqueeze(0) ile batch boyutu ekle
    input_tensor = transform(image).unsqueeze(0).to(device)

    # 7. Modele ver (forward pass) ve tahminleri al
    with torch.no_grad(): # Tahmin yaparken gradyan hesaplamaya gerek yok
        output = model(input_tensor)
        # torch.argmax kullanarak en yüksek olasılığa sahip sınıf indekslerini (0 veya 1) alma
        predicted_classes = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # 8. Sınıf indekslerini görselleştirmek için tekrar RGB renklerine çevirme
    prediction_visual = np.zeros((predicted_classes.shape[0], predicted_classes.shape[1], 3), dtype=np.uint8)
    for idx, color in INDEX_TO_COLOR.items():
        prediction_visual[predicted_classes == idx] = color

    # 9. Sonucu kaydetme
    result_image = Image.fromarray(prediction_visual)
    save_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(save_dir, "tahmin_sonucu.png")
    
    result_image.save(save_path)
    print(f"Tahmin sonucu başarıyla kaydedildi: {save_path}")

if __name__ == "__main__":
    main()
