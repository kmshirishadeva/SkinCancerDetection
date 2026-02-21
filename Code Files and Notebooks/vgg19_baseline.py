# vgg19_baseline.py
import os, random, numpy as np, torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ======= CONFIG =======
data_dir = r"D:\Major Project\archive\data"
train_dir = os.path.join(data_dir, "train")
test_dir  = os.path.join(data_dir, "test")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed, batch_size, epochs, lr = 42, 16, 8, 1e-4
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

# ======= AUGMENTATION =======
def get_tfms(train=True, size=224):
    if train:
        return A.Compose([
            A.RandomResizedCrop(size=(size, size), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            A.Normalize(),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(height=size, width=size),
            A.Normalize(),
            ToTensorV2()
        ])

class AlbDataset(Dataset):
    def __init__(self, folder, transform):
        self.ds = datasets.ImageFolder(folder)
        self.tf = transform
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        img, label = self.ds[i]
        img = np.array(img)
        img = self.tf(image=img)['image']
        return img, label

# ======= MODEL =======
class VGG19Binary(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = timm.create_model('vgg19', pretrained=True)
        n_features = self.base.get_classifier().in_features
        self.base.reset_classifier(num_classes=0)
        self.classifier = nn.Linear(n_features, 1)
    def forward(self, x):
        x = self.base(x)
        return self.classifier(x).squeeze(1)

# ======= TRAIN & EVAL =======
def train_epoch(model, loader, opt, loss_fn):
    model.train(); total=0
    for x, y in loader:
        x, y = x.to(device).float(), y.to(device).float()
        opt.zero_grad(); out = model(x); loss = loss_fn(out, y)
        loss.backward(); opt.step()
        total += loss.item()*x.size(0)
    return total/len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); preds, labels = [], []
    for x, y in loader:
        x = x.to(device).float()
        out = torch.sigmoid(model(x)).cpu().numpy()
        preds.extend(out); labels.extend(y.numpy())
    preds_bin = (np.array(preds)>0.5).astype(int)
    return {
        "acc": accuracy_score(labels, preds_bin),
        "prec": precision_score(labels, preds_bin),
        "rec": recall_score(labels, preds_bin),
        "f1": f1_score(labels, preds_bin),
        "auc": roc_auc_score(labels, preds)
    }

# ======= MAIN =======
def main():
    train_dl = DataLoader(AlbDataset(train_dir,get_tfms(True)),batch_size=batch_size,shuffle=True,num_workers=2)
    test_dl  = DataLoader(AlbDataset(test_dir,get_tfms(False)),batch_size=batch_size,shuffle=False,num_workers=2)

    model = VGG19Binary().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc = 0
    for ep in range(1, epochs+1):
        tl = train_epoch(model, train_dl, opt, loss_fn)
        metrics = evaluate(model, test_dl)
        print(f"Epoch {ep}: Loss={tl:.4f}, AUC={metrics['auc']:.4f}, ACC={metrics['acc']:.4f}, F1={metrics['f1']:.4f}")
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            torch.save(model.state_dict(), "vgg19_baseline.pth")

    print("\nFinal Metrics:")
    print(metrics)

if __name__ == "__main__":
    main()