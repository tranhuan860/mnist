import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
import random
import numpy as np
def random_erase_4x4(img, erase_value=0):
    """
    img: torch.Tensor of shape [C, H, W]
    erase_value: value to fill (default 0)
    """
    C, H, W = img.shape
    
    patch_h, patch_w = 4, 4  # size of erase region
    if H < patch_h or W < patch_w:
        return img  # skip if image too small
    
    # choose random top-left corner
    top = random.randint(0, H - patch_h)
    left = random.randint(0, W - patch_w)
    
    # zero out the patch
    img[:, top:top+patch_h, left:left+patch_w] = erase_value
    return img

class RandomErase4x4:
    def __init__(self, p=0.5, erase_value=0):
        self.p = p
        self.erase_value = erase_value
        
    def __call__(self, img):
        if random.random() < self.p:
            return random_erase_4x4(img, self.erase_value)
        return img


def loadMNIST(
    data_dir='./data',
    batch_size=64,
    num_workers=2,
    pin_memory=True,
    val_ratio=0.1
):
    """
    Tải MNIST và trả về train_loader, val_loader, test_loader.
    - data_dir: thư mục lưu MNIST
    - batch_size: batch size cho train/val/test
    - num_workers: số worker cho DataLoader
    - pin_memory: pin_memory cho DataLoader
    - val_ratio: tỷ lệ validation so với train
    """
    # 1. Transform
    train_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(
            degrees=0, translate=(0.1, 0.1)      # Dịch tối đa 10% theo x, y
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        RandomErase4x4(p=0.5, erase_value=0)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 2. Dataset gốc train (có augmentation)
    full_train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=train_transform)

    # Dataset thứ 2 cho val (không augmentation)
    val_full_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=val_transform)

    # Dataset test
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=test_transform)

    # 3. Stratified split cho train/val
    targets = full_train_dataset.targets.numpy()
    train_idx, val_idx = train_test_split(
        range(len(targets)),
        test_size=val_ratio,
        random_state=42,
        stratify=targets
    )

    # 4. Tạo Subset
    train_dataset = Subset(full_train_dataset, train_idx)
    val_dataset = Subset(val_full_dataset, val_idx)

    # 5. DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    # 6. In ra độ dài
    print(f"Train: {len(train_dataset)}")
    print(f"Val: {len(val_dataset)}")
    print(f"Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader, ConcatDataset
import numpy as np
from torchvision import datasets, transforms
from collections import defaultdict
import torch, os
class FER2013Dataset(Dataset):
    def __init__(self, csv_path, usage="Training", transform=None):
        data = pd.read_csv(csv_path)
        self.data = data[data["Usage"] == usage].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        emotion = int(row["emotion"])
        pixels = np.fromstring(row["pixels"], sep=" ", dtype=np.uint8).reshape(48, 48)
        img = Image.fromarray(pixels)

        if self.transform:
            img = self.transform(img)
        return img, emotion

def loadFER2013(
    data_dir='./data',
    batch_size=64,
    num_workers=2,
    pin_memory=True,
    **kwargs
):
    # Sample transforms
    transform_train = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomResizedCrop(48, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # transforms.RandAugment(num_ops=2, magnitude=6),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                  std=[0.229, 0.224, 0.225])
        # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
    ])

    transform_val = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                  std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_ds = FER2013Dataset(os.path.join(data_dir, "fer2013/fer2013.csv"), usage="Training", transform=transform_train)
    val_ds   = FER2013Dataset(os.path.join(data_dir, "fer2013/fer2013.csv"), usage="PublicTest", transform=transform_val)
    test_ds  = FER2013Dataset(os.path.join(data_dir, "fer2013/fer2013.csv"), usage="PrivateTest", transform=transform_val)
    # # =========================
    # # Lấy nhãn
    # # =========================
    # targets = np.array([sample[1] for sample in train_ds])

    # # =========================
    # # Oversample để train dataset cân bằng
    # # =========================
    # target_count = 6000  # số lượng ảnh mong muốn mỗi nhãn
    # class_indices = defaultdict(list)

    # # Gom index của từng nhãn
    # for idx, label in enumerate(targets):
    #     class_indices[label].append(idx)

    # balanced_train_datasets = []

    # for label, indices in class_indices.items():
    #     current_count = len(indices)
    #     if current_count < target_count:
    #         # Oversample + augmentation
    #         extra_indices = np.random.choice(indices, target_count - current_count)
    #         all_indices = indices + list(extra_indices)
    #     else:
    #         # Nếu nhiều hơn, undersample
    #         all_indices = np.random.choice(indices, target_count, replace=False)
        
    #     subset = Subset(train_ds, all_indices)
    #     balanced_train_datasets.append(subset)

    # # Kết hợp tất cả nhãn thành 1 dataset cân bằng
    # balanced_train_ds = ConcatDataset(balanced_train_datasets)
    # train_ds = balanced_train_ds
    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # Sanity check
    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}, Test size: {len(test_ds)}")
    return train_loader, val_loader, test_loader
