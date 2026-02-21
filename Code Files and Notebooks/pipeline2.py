# train_pipeline.py
import os
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# ---------- Config ----------
data_dir = r"D:\Major Project\archive\data"                # root folder containing ‘train’ and ‘test’
train_dir = os.path.join(data_dir, "train")
test_dir  = os.path.join(data_dir, "test")
backbone = "efficientnet_b3"
seed     = 42
batch_size = 16
epochs     = 8
lr         = 1e-4
embed_dim  = 512
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seeds
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# ---------- Transforms ----------
def get_alb_transforms(train=True, img_size=224):
    if train:
        return  A.Compose([
                 A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
                 A.HorizontalFlip(p=0.5),
                 A.VerticalFlip(p=0.2),
                 A.RandomBrightnessContrast(0.2,0.2,p=0.5),
                 A.Rotate(limit=90, p=0.5),
                 A.ElasticTransform(p=0.2),
                 A.HueSaturationValue(p=0.2),
                 A.CLAHE(p=0.3),
                 A.Normalize(),
                 ToTensorV2()
                ])
    else:
        return A.Compose([
            A.Resize(height=img_size, width=img_size),  # Resize still uses height/width
            A.Normalize(),
            ToTensorV2()
        ])

class AlbumentationsDataset(Dataset):
    def __init__(self, image_dataset: datasets.ImageFolder, transform):
        self.image_dataset = image_dataset
        self.transform       = transform

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        img, label = self.image_dataset[idx]
        img        = np.array(img)
        img        = self.transform(image=img)["image"]
        return img, label

# ---------- Attention & Model ----------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x)
        return x * w

class AttentionBackbone(nn.Module):
    def __init__(self, backbone_name='resnet34', pretrained=True, embed_dim=512):
        super().__init__()
        self.backbone   = timm.create_model(backbone_name, pretrained=pretrained, features_only=True)
        feat_channels   = self.backbone.feature_info[-1]['num_chs']
        self.se         = SEBlock(feat_channels)
        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.fc         = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_channels, embed_dim),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x, return_feats=False):
        feats  = self.backbone(x)[-1]
        feats  = self.se(feats)
        pooled = self.pool(feats)
        emb    = self.fc(pooled)
        logits = self.classifier(emb).squeeze(1)
        if return_feats:
            return emb
        return logits

# ---------- Training & Validating ----------
def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs   = imgs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)
        opt.zero_grad()
        logits = model(imgs)
        loss   = loss_fn(logits, labels)
        loss.backward()
        opt.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    preds, reals = [], []
    for imgs, labels in loader:
        imgs   = imgs.to(device, dtype=torch.float)
        logits = model(imgs)
        probs  = torch.sigmoid(logits).cpu().numpy()
        preds.extend(probs.tolist())
        reals.extend(labels.numpy().tolist())
    if len(set(reals)) > 1:
        auc = roc_auc_score(reals, preds)
    else:
        auc = 0.5
    return auc, preds, reals

@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    emb_list, label_list = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, dtype=torch.float)
        emb  = model(imgs, return_feats=True)
        emb_list.append(emb.cpu().numpy())
        label_list.append(labels.numpy())
    return np.vstack(emb_list), np.concatenate(label_list)

# ---------- Main ----------
def main():
    # create datasets
    train_dataset_raw = datasets.ImageFolder(train_dir)
    test_dataset_raw  = datasets.ImageFolder(test_dir)

    train_ds = AlbumentationsDataset(train_dataset_raw, get_alb_transforms(True))
    test_ds  = AlbumentationsDataset(test_dataset_raw, get_alb_transforms(False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # model
    model   = AttentionBackbone(backbone_name=backbone, pretrained=True, embed_dim=embed_dim).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    opt     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_auc = 0.0
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, opt, loss_fn, device)
        val_auc, _, _ = validate(model, test_loader, device)
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} — train_loss: {train_loss:.4f}, val_auc: {val_auc:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "model2.pth")

    # embeddings + XGBoost
    model.load_state_dict(torch.load("model2.pth"))
    train_emb, train_lbl = extract_embeddings(model, train_loader, device)
    test_emb,  test_lbl  = extract_embeddings(model, test_loader, device)

    dtrain = xgb.DMatrix(train_emb, label=train_lbl)
    dtest  = xgb.DMatrix(test_emb,  label=test_lbl)
    params = {
        "objective":        "binary:logistic",
        "eval_metric":      "auc",
        "learning_rate":    0.05,
        "max_depth":        6,
        "subsample":        0.8,
        "colsample_bytree": 0.7,
        "seed":             seed
    }
    booster = xgb.train(params, dtrain, num_boost_round=200, evals=[(dtest, "test")], early_stopping_rounds=20)
    pred   = booster.predict(dtest)
    print("XGBoost Test AUC:", roc_auc_score(test_lbl, pred))
    print("XGBoost Test Acc:", accuracy_score(test_lbl, (pred>0.5).astype(int)))
    print("XGBoost Test Prec:", precision_score(test_lbl, (pred>0.5).astype(int)))
    print("XGBoost Test Rec:", recall_score(test_lbl, (pred>0.5).astype(int)))
    print("XGBoost Test F1:", f1_score(test_lbl, (pred>0.5).astype(int)))

if __name__ == "__main__":
    main()