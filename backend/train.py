import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import segmentation_models_pytorch as smp

# Örnek renk haritası (Kendi veri setinizin RGB renklerine göre güncelleyin)
# Veri setinizdeki gerçek RGB renklerine göre güncellenmiş renk haritası
# Format: (R, G, B): Sınıf İndeksi
COLOR_MAP = {
    (0, 0, 0): 0,
    (128, 0, 0): 1,
    (0, 128, 0): 2,
    (128, 128, 0): 3,
    (64, 0, 128): 4,
    (192, 0, 192): 5,
    (64, 64, 0): 6,
    (128, 64, 128): 7,
}
NUM_CLASSES = len(COLOR_MAP)

def rgb_to_mask(img_array, color_map):
    """RGB maskeyi, color_map'e göre 2D sınıf indeks (0, 1, 2...) matrisine çevirir."""
    mask = np.zeros(img_array.shape[:2], dtype=np.int64)
    for rgb, idx in color_map.items():
        # img_array içerisindeki her bir pikselin rgb ile eşleşip eşleşmediğini kontrol et
        match = (img_array == rgb).all(axis=-1)
        mask[match] = idx
    return mask

class DroneDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        if os.path.exists(root_dir):
            # seq1, seq2 gibi klasörleri bul
            seq_folders = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
            
            for seq in seq_folders:
                images_dir = os.path.join(seq, "Images")
                labels_dir = os.path.join(seq, "Labels")
                
                if os.path.exists(images_dir) and os.path.exists(labels_dir):
                    for img_name in os.listdir(images_dir):
                        if img_name.endswith(('.png', '.jpg', '.jpeg', '.tif')):
                            img_path = os.path.join(images_dir, img_name)
                            mask_path = os.path.join(labels_dir, img_name)
                            
                            if os.path.exists(mask_path):
                                self.samples.append((img_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB") # RGB maske renklerini kaybetmemek için RGB olarak açıyoruz
        
        if self.transform:
            image = self.transform(image)
            mask_resize = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST)
            mask = mask_resize(mask)
        else:
            image = transforms.ToTensor()(image)
        # Maske piksellerini RGB'den sınıf indekslerine (0, 1, 2...) çevirme
        mask_array = np.array(mask)
        mask_idx = rgb_to_mask(mask_array, COLOR_MAP)
        mask = torch.from_numpy(mask_idx).long()
        return image, mask



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Eğitim veri setinin kök dizini", default="archive/train/train")
    args = parser.parse_args()

    # 1. Dataset (DroneDataset) dosya yolları
    # backend dizininden çalıştırıldığı varsayımıyla, ana dizindeki archive'e erişim.
    train_root_dir = args.data_dir
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    dataset = DroneDataset(root_dir=train_root_dir, transform=transform)
    
    # 2. DataLoader (VRAM 6GB olduğu için batch_size=4)
    # Eğer dizinler yoksa veya boşsa hata almamak için kontrol ekliyoruz.
    if len(dataset) == 0:
        print("Dataset boş veya klasör bulunamadı, dummy verilerle eğitim çalıştırılıyor...")
        # dummy veriler ekleyebiliriz ama bu scriptin şablonu yeterli. 
        # Yine de kodun sonlanmaması için dummy DataLoader oluşturalım:
        dummy_images = torch.randn(8, 3, 256, 256)
        dummy_masks = torch.zeros(8, 256, 256, dtype=torch.long)
        dummy_dataset = torch.utils.data.TensorDataset(dummy_images, dummy_masks)
        dataloader = DataLoader(dummy_dataset, batch_size=4, shuffle=True)
    else:
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")
    
    # Modelin yüklenmesi (Sınıf sayısını dinamik yapıyoruz)
    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=NUM_CLASSES).to(device)
    
    # 3. Kayıp fonksiyonu (DiceLoss) ve Optimizer (Adam)
    criterion = smp.losses.DiceLoss(mode='multiclass')
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 4. 50 epoch'luk eğitim döngüsü
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train() # Modeli eğitim moduna alıyoruz
        running_loss = 0.0
        
        for images, masks in dataloader:
            # Batch'leri cihaz (cuda) üzerine gönder
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Kayıp hesaplama
            loss = criterion(outputs, masks)
            
            # Geri yayılım (Backward) ve optimizasyon
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # Her epoch sonunda ortalama kaybı hesaplayıp terminale yazdırma
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        
    # 5. Ağırlıkları 'best_model.pth' olarak backend/ dizinine kaydetme
    # main dosyası doğrudan çalıştırıldığında (train.py ile) mevcut dizine (backend/) kaydedecektir.
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Eğitim tamamlandı! Model ağırlıkları kaydedildi: {save_path}")

if __name__ == "__main__":
    main()
