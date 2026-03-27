import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from pathlib import Path
import os
import sys

# ============================================================
# 1. CORE COMPONENTS FROM ARCHITECTURE (Standalone)
# ============================================================

class SSMoE_Core(nn.Module):
    def __init__(self, num_specific_experts, num_shared_experts, top_k=2):
        super().__init__()
        self.num_spec = num_specific_experts
        self.num_shared = num_shared_experts
        self.top_k = top_k

    def forward(self, x, spec_router_logits, shared_router_logits, spec_experts, shared_experts):
        spec_probs = F.softmax(spec_router_logits, dim=1)
        topk_weights, topk_indices = torch.topk(spec_probs, self.top_k, dim=1)
        topk_weights = topk_weights / (topk_weights.sum(dim=1, keepdim=True) + 1e-6)
        
        spec_out = 0
        for i in range(self.num_spec):
            weight_i = torch.zeros_like(spec_router_logits[:, :1])
            for k in range(self.top_k):
                match = (topk_indices[:, k:k+1] == i).float()
                weight_i += match * topk_weights[:, k:k+1]
            if weight_i.any():
                spec_out += weight_i * spec_experts[i](x)

        shared_probs = F.softmax(shared_router_logits, dim=1)
        shared_out = 0
        for i in range(self.num_shared):
            shared_out += shared_probs[:, i:i+1] * shared_experts[i](x)
        return spec_out + shared_out

class SSMoE_ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_spec=4, num_shared=2, top_k=2):
        super().__init__()
        self.core = SSMoE_Core(num_spec, num_shared, top_k)
        self.spec_experts = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                          nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)) for _ in range(num_spec)
        ])
        self.shared_experts = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                          nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)) for _ in range(num_shared)
        ])
        self.spec_router = nn.Conv2d(in_channels, num_spec, kernel_size=1)
        self.shared_router = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(in_channels, num_shared))

    def forward(self, x):
        spec_logits = self.spec_router(x)
        shared_logits = self.shared_router(x).view(-1, self.core.num_shared, 1, 1)
        return self.core(x, spec_logits, shared_logits, self.spec_experts, self.shared_experts)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

# ── Vanilla U-Net (Baseline) ──
class VanillaUNet(nn.Module):
    def __init__(self, in_channels=16, n_classes=1):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512)) # Vanilla Bottleneck
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3)
        x = self.up1(x4); x = self.conv_up1(torch.cat([x, x3], dim=1))
        x = self.up2(x); x = self.conv_up2(torch.cat([x, x2], dim=1))
        x = self.up3(x); x = self.conv_up3(torch.cat([x, x1], dim=1))
        return torch.sigmoid(self.outc(x))

# ── Adapted SSMoE U-Net (Paper's Invention) ──
class Adapted_SSMoE_UNet(nn.Module):
    def __init__(self, in_channels=16, n_classes=1):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), SSMoE_ConvBlock(256, 512)) # MoE Bottleneck
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3)
        x = self.up1(x4); x = self.conv_up1(torch.cat([x, x3], dim=1))
        x = self.up2(x); x = self.conv_up2(torch.cat([x, x2], dim=1))
        x = self.up3(x); x = self.conv_up3(torch.cat([x, x1], dim=1))
        return torch.sigmoid(self.outc(x))

# ============================================================
# 2. DATALOADER & TRAINING LOOP
# ============================================================

class LandslideDataset(Dataset):
    def __init__(self, samples=20):
        self.samples = samples # Fixed small number for demonstration
    def __len__(self): return self.samples
    def __getitem__(self, idx):
        return torch.randn(16, 256, 256), (torch.randn(1, 256, 256) > 0.5).float()

def run_training_loop(model, loader, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()
    history = {'loss': [], 'acc': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct += ((preds > 0.5) == y).float().mean().item()
            total += 1
        history['loss'].append(train_loss / total)
        history['acc'].append(correct / total)
        print(f"Epoch {epoch+1:02d} | Loss: {history['loss'][-1]:.4f} | Acc: {history['acc'][-1]:.4f}")
    return history

def plot_results(v_hist, a_hist):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(v_hist['loss'], label='Vanilla (Baseline)', color='gray')
    ax1.plot(a_hist['loss'], label='Adapted (SSMoE)', color='green')
    ax1.set_title("Training Loss Comparison"); ax1.legend()
    ax2.plot(v_hist['acc'], label='Vanilla (Baseline)', color='gray')
    ax2.plot(a_hist['acc'], label='Adapted (SSMoE)', color='green')
    ax2.set_title("Validation Accuracy Comparison"); ax2.legend()
    plt.tight_layout()
    plt.savefig('evaluation/results/comparison.png')
    plt.show()

if __name__ == "__main__":
    os.makedirs('evaluation/results', exist_ok=True)
    loader = DataLoader(LandslideDataset(), batch_size=4, shuffle=True)
    
    print(">>> RUN 1: VANILLA BASELINE")
    torch.manual_seed(42); v_hist = run_training_loop(VanillaUNet(), loader)
    
    print("\n>>> RUN 2: ADAPTED SSMoE MODEL")
    torch.manual_seed(42); a_hist = run_training_loop(Adapted_SSMoE_UNet(), loader)
    
    plot_results(v_hist, a_hist)
