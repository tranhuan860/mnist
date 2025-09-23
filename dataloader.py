import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader

def load(
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
        transforms.RandomRotation(10),
        transforms.RandomAffine(
            degrees=0, translate=(0.1, 0.1)      # Dịch tối đa 10% theo x, y
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
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
