from typing import Tuple
import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision.datasets import MNIST
from torchvision import transforms
from PIL import Image
import os

import glob
import torchvision.transforms as transforms

def rand_dataset(num_rows=60_000, num_columns=100) -> Dataset:
    return TensorDataset(torch.rand(num_rows, num_columns))


def mnist_dataset(train=True) -> Dataset:
    """
    Returns the MNIST dataset for training or testing.
    
    Args:
    train (bool): If True, returns the training dataset. Otherwise, returns the testing dataset.
    
    Returns:
    Dataset: The MNIST dataset.
    """
    return MNIST(root='./data', train=train, download=True, transform=None)

class RadarImageDataset(Dataset):
    """
    Dataset für Radar-Bilder mit Anomaly Detection
    """
    
    def __init__(self, image_paths, labels=None, transform=None, image_size=(128, 128)):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.image_size = image_size
        
        # Standard Transform falls keiner gegeben
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])  # Normalisierung für Grayscale
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Bild laden
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path)
            
            # Konvertiere zu RGB falls nötig, dann zu Grayscale für Konsistenz
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = image.convert('L')  # Grayscale
            
            # Transform anwenden
            image = self.transform(image)
            
            if self.labels is not None:
                return image, self.labels[idx]
            else:
                return image
                
        except Exception as e:
            print(f"Fehler beim Laden von {img_path}: {e}")
            # Fallback: schwarzes Bild
            image = torch.zeros(1, *self.image_size)
            if self.labels is not None:
                return image, self.labels[idx]
            else:
                return image

def load_radar_data(data_dir: str, image_size: int = 128):
    """
    Lädt Radar-Bilddaten aus dem angegebenen Verzeichnis
    
    Dateiname Format: <ExpID>P<SubjectID>A<ActivityID>R<RunID>_<SegmentIndex>_<Modus>[_mf].png
    A01-A05 = normale Aktivitäten
    A06 = Fallen (Anomalie)
    
    Args:
        data_dir: Pfad zum Datenverzeichnis
        image_size: Größe für Bildresize
    
    Returns:
        normal_paths, anomaly_paths: Listen der Bildpfade
    """
    # Alle PNG-Dateien finden
    
    all_files = glob.glob(os.path.join(data_dir, "*.png"))
    #print("all_files", all_files)
    normal_paths = []
    anomaly_paths = []
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        try:
            # Suche nach Activity ID Pattern "A01" bis "A06"
            import re
            activity_match = re.search(r'A(\d{2})', filename)
            #print("Ziffer", activity_match)
            if activity_match:
                activity_id = int(activity_match.group(1))
                
                if activity_id in [1, 2, 3, 4, 5]:  # A01 bis A05 = normale Aktivitäten
                    normal_paths.append(file_path)
                elif activity_id == 6:  # A06 = Fallen (Anomalie)
                    anomaly_paths.append(file_path)
                else:
                    print(f"Unbekannte Activity ID A{activity_id:02d} in Datei: {filename}")
            else:
                print(f"Keine Activity ID gefunden in Datei: {filename}")
                
        except Exception as e:
            print(f"Fehler beim Parsen der Datei {filename}: {e}")
    
    print(f"Gefunden: {len(normal_paths)} normale Bilder (A01-A05), {len(anomaly_paths)} Anomalie-Bilder (A06)")
    
    # Zeige Beispiele der gefundenen Dateien
    if normal_paths:
        print(f"Beispiel normale Datei: {os.path.basename(normal_paths[0])}")
    if anomaly_paths:
        print(f"Beispiel Anomalie-Datei: {os.path.basename(anomaly_paths[0])}")
    
    return normal_paths, anomaly_paths


def radar_dataset(image_size: int = 128): #data_dir_train: str, data_dir_val: str, 
    """
    Trainiert VAE auf Radar-Daten für Anomaly Detection
    """
    print("Lade Radar-Daten...")
    
    data_dir_train = r"Link\to\radar_dataset\train"
    data_dir_test = r"Link\to\radar_dataset\test"
    # Daten laden
    normal_paths, anomaly_paths = load_radar_data(data_dir_train, image_size)

    if len(normal_paths) == 0:
        raise ValueError("Keine normalen Bilder gefunden! Überprüfe den Datenpfad und Dateinamen.")
    
    # Transforms für Datenaugmentation
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Datasets erstellen - nur normale Daten für Training
    train_dataset = RadarImageDataset(normal_paths, transform=train_transform, image_size=(image_size, image_size))
    
    # Test Dataset mit normalen und anomalen Daten
    all_val_paths = normal_paths + anomaly_paths
    val_labels = [0] * len(normal_paths) + [1] * len(anomaly_paths)  # 0 = normal, 1 = anomaly
    val_dataset = RadarImageDataset(all_val_paths, labels=val_labels, transform=test_transform, image_size=(image_size, image_size))

    # Lade Daten für das Testset und erstelle das Test-Dataset
    test_normal_paths, test_anomaly_paths = load_radar_data(data_dir_test, image_size)
    all_test_paths = test_normal_paths + test_anomaly_paths
    test_labels = [0] * len(test_normal_paths) + [1] * len(test_anomaly_paths)
    test_dataset = RadarImageDataset(all_test_paths, labels=test_labels, transform=test_transform, image_size=(image_size, image_size))
    return train_dataset, val_dataset, test_dataset