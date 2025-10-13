# ==== IMPORTS ==== #
import random
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Subset
from torch.nn.functional import softmax
from torchvision import datasets, transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.amp import autocast, GradScaler


# ==== TRAINING ==== #
def training(
    model,
    train_loader,
    val_loader,
    epochs,
    optimizer,
    scheduler=None,
    device=None,
    criterion=None,
    resultF=None,
    continues=None,
    patience=5,       # số epoch không cải thiện để dừng
    delta=1e-4        # mức cải thiện tối thiểu
):
    best_acc = 0
    st_epoch = 0
    wait = 0  # đếm số epoch không cải thiện liên tục

    # ---- HISTORY ---- #
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    if continues:
        checkpoint = torch.load(resultF, map_location=device)
        for i in range(len(model)):
          model[i].load_state_dict(checkpoint['model_state_dict'][i])
        best_acc = checkpoint["val_acc"]
        st_epoch = checkpoint['epoch']
        if 'history' in checkpoint:   # nếu có lưu history trước đó
            history = checkpoint['history']

    scaler = GradScaler(device=device)  # AMP scaler

    for epoch in range(st_epoch + 1, epochs + 1):
        for i in range(len(model)): 
            model[i].train()
        train_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch")
        for x, y in pbar:
            x = x.to(device, memory_format=torch.channels_last)
            y = y.squeeze().long().to(device)

            with autocast(device_type='cuda'):
                outs = []
                feats = []
                for i in range(len(model)): 
                  out, feat = model[i](x)
                  outs.append(out)
                  feats.append(feat)
                out = sum(outs)/len(outs)
                loss = criterion(out, y)
                losss = [criterion(outs[i], y) for i in range(len(model))]
            
            for i in range(len(model)):
              optimizer[i].zero_grad()
            losss_cpu = [i.detach().cpu().item() for i in losss]
            cos_loss = F.mse_loss(feats[np.argmax(losss_cpu)], feats[np.argmin(losss_cpu)])
            scaler.scale(loss + cos_loss * 0.1).backward()
            scaler.step(optimizer[np.argmax(losss_cpu)])

            scaler.update()
            if scheduler:
                scheduler.step()

            train_loss += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == y).sum()
            total += y.size(0)

            if scheduler:
                pbar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])

        train_loss /= len(train_loader)
        train_acc = correct.item() / total

        # ---- VALIDATION ---- #
        for i in range(len(model)): 
            model[i].eval()
        val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad(), autocast(device_type='cuda'):
            for x, y in val_loader:
                x = x.to(device, memory_format=torch.channels_last)
                y = y.squeeze().long().to(device)
                outs = []
                feats = []
                for i in range(len(model)): 
                  out, feat = model[i](x)
                  outs.append(out)
                  feats.append(feat)
                out = sum(outs)/len(outs)
                loss = criterion(out, y)
                val_loss += loss.item()

                preds = out.argmax(dim=1)
                correct += (preds == y).sum()
                total += y.size(0)

        val_loss /= len(val_loader)
        val_acc = correct.item() / total

        # ---- LƯU HISTORY ---- #
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # ---- Early Stopping logic ---- #
        if val_acc > best_acc + delta:
            best_acc = val_acc
            wait = 0
            # Lưu checkpoint
            model_state_dict = [model[i].state_dict() for i in range(len(model))]
            optimizer_state_dict = [optimizer[i].state_dict() for i in range(len(model))]
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer_state_dict,
                'scaler_state_dict': scaler.state_dict(),
                'val_acc': val_acc,
                'history': history
            }, resultF)
            print(f"Saved best model at epoch {epoch} with Val Acc: {val_acc:.4f}")
        else:
            wait += 1
            print(f"No improvement for {wait} epoch(s). Patience left: {patience - wait}")

        print(f"Epoch {epoch}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if wait >= patience:
            print(f"Early stopping at epoch {epoch}. Best Val Acc: {best_acc:.4f}")
            break

# ==== EVALUATION ==== #
def evaluate(
    model,
    class_names,
    test_loader,
    device=None,
    resultF=None
):
    # 1. Load checkpoint
    checkpoint = torch.load(resultF, map_location=device)
    for i in range(len(model)):
        model[i].load_state_dict(checkpoint['model_state_dict'][i])
    for i in range(len(model)):
        model[i].eval()

    # Nếu có history thì lấy ra
    history = checkpoint.get('history', None)
    if history is not None:
        # Vẽ biểu đồ Loss
        plt.figure(figsize=(8,6))
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training & Validation Loss')
        plt.legend()
        plt.grid()
        plt.show()

        # Vẽ biểu đồ Accuracy
        plt.figure(figsize=(8,6))
        plt.plot(history['train_acc'], label='Train Acc')
        plt.plot(history['val_acc'], label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training & Validation Accuracy')
        plt.legend()
        plt.grid()
        plt.show()

    correct, total = 0, 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad(), autocast(device_type='cuda'):
        for x, y in tqdm(test_loader, unit="batch"):
            x = x.to(device, memory_format=torch.channels_last)
            y = y.long().to(device)
            outs = []
            feats = []
            for i in range(len(model)): 
                out, feat = model[i](x)
                outs.append(out)
                feats.append(feat)
            out = sum(outs)/len(outs)

            preds = out.argmax(dim=1)
            correct += (preds == y).sum()
            total += y.size(0)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            probs = softmax(out, dim=1)  # shape [batch_size, num_classes]
            all_probs.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    # 4. Tính các chỉ số tổng hợp
    acc = accuracy_score(all_labels, all_preds)
    precision_macro = precision_score(all_labels, all_preds, average='macro')
    recall_macro    = recall_score(all_labels, all_preds, average='macro')
    f1_macro        = f1_score(all_labels, all_preds, average='macro')

    # 5. Tính từng nhãn
    precision_per_class = precision_score(all_labels, all_preds, average=None)
    recall_per_class    = recall_score(all_labels, all_preds, average=None)
    f1_per_class        = f1_score(all_labels, all_preds, average=None)

    # 6. In kết quả
    print("=== Kết quả tổng hợp ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro Precision: {precision_macro:.4f}")
    print(f"Macro Recall: {recall_macro:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")

    print("\n=== Kết quả cho từng lớp ===")
    for i, cls in enumerate(class_names):
        print(f"Lớp {cls}: Precision={precision_per_class[i]:.4f}, "
              f"Recall={recall_per_class[i]:.4f}, "
              f"F1={f1_per_class[i]:.4f}")

    print("\n=== Classification Report ===")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion matrix raw counts
    cm = confusion_matrix(all_labels, all_preds)

    # --- CHUẨN HÓA THEO HÀNG ---
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # tỷ lệ theo hàng

    plt.figure(figsize=(8,6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix (per row)')
    plt.show()

    n_classes = len(class_names)

    # Binarize labels (one-hot)
    y_test_bin = label_binarize(all_labels, classes=list(range(n_classes)))  
    y_score = all_probs  # xác suất dự đoán

    # Vẽ ROC cho từng lớp
    plt.figure(figsize=(8,6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {class_names[i]} (AUC = {roc_auc:.2f})')

    # Vẽ đường chéo
    plt.plot([0,1], [0,1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (One-vs-Rest)')
    plt.legend()
    plt.show()

