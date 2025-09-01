import torch
from ultralytics import YOLO

def train_model():
    # Modeli yükleyin
    model = YOLO("yolo11n.pt")
    
    # GPU kullanılabilirliği kontrolü
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")
    
    # Modeli eğitin9
    results = model.train(data="C:\\Users\\Beyza\\source\\uav_model\\uav.yaml", epochs=10, imgsz=640, device=device)

if __name__ == '__main__':
    train_model()
