# DOMAIN TRANSLATION: EEG 1D (B, C, S) -> Geospatial CV 2D (B, C, H, W)
# Paper: EEGMoE: A Domain-Decoupled Mixture-of-Experts Model for Self-Supervised EEG Representation Learning
# Core invention: Specific and Shared Mixture-of-Experts (SSMoE) with dual routing.

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# PAPER'S CORE INVENTION — DO NOT MODIFY
# [Eq. 1-5] Specific and Shared Mixture-of-Experts (SSMoE)
# ============================================================
class SSMoE_Core(nn.Module):
    """
    Core Logic for Specific and Shared Mixture-of-Experts.
    Preserves the mathematical decoupling of domain-specific (Top-K) 
    and domain-shared (Soft) representations.
    """
    def __init__(self, num_specific_experts, num_shared_experts, top_k=2):
        super().__init__()
        self.num_spec = num_specific_experts
        self.num_shared = num_shared_experts
        self.top_k = top_k

    def forward(self, x, spec_router_logits, shared_router_logits, spec_experts, shared_experts):
        # 1. Specific MoE: Top-K Routing [Eq. 3]
        # x shape: (B, C, H, W) or (B, N, D)
        # router_logits shape: (B, num_spec, ...)
        
        # Sparse routing logic
        spec_probs = F.softmax(spec_router_logits, dim=1)
        topk_weights, topk_indices = torch.topk(spec_probs, self.top_k, dim=1)
        
        # Normalize top-k weights [Eq. 2]
        topk_weights = topk_weights / (topk_weights.sum(dim=1, keepdim=True) + 1e-6)
        
        # Apply specific experts
        # Note: In 2D, we apply experts locally. We aggregate outputs using the router map.
        spec_out = 0
        for i in range(self.num_spec):
            # Mask to select only tokens/pixels where expert i is in Top-K
            # Indices match i across any of the top_k channels
            # weight_i shape: (B, 1, H, W)
            weight_i = torch.zeros_like(spec_router_logits[:, :1])
            for k in range(self.top_k):
                match = (topk_indices[:, k:k+1] == i).float()
                weight_i += match * topk_weights[:, k:k+1]
            
            # Efficiently compute expert output only if it's selected anywhere in the batch
            if weight_i.any():
                spec_out += weight_i * spec_experts[i](x)

        # 2. Shared MoE: Soft Routing [Eq. 4]
        shared_probs = F.softmax(shared_router_logits, dim=1)
        shared_out = 0
        for i in range(self.num_shared):
            # shared_probs[:, i:i+1] is (B, 1, ...)
            shared_out += shared_probs[:, i:i+1] * shared_experts[i](x)
            
        # 3. Final Fusion [Eq. 5]
        return spec_out + shared_out

# ============================================================
# ADAPTED COMPONENT: SSMoE_ConvBlock
# Translates 1D MLP Experts to 2D Conv Experts
# ============================================================
class SSMoE_ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_spec=4, num_shared=2, top_k=2):
        super().__init__()
        self.core = SSMoE_Core(num_spec, num_shared, top_k)
        
        # SWAP: Linear Experts -> 3x3 Conv2d
        self.spec_experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(num_spec)
        ])
        
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(num_shared)
        ])
        
        # SWAP: Linear Router -> 1x1 Conv2d (Local Routing for Specific)
        self.spec_router = nn.Conv2d(in_channels, num_spec, kernel_size=1)
        
        # SWAP: Linear Router -> GAP + MLP (Global Routing for Shared)
        self.shared_router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_shared)
        )

    def forward(self, x):
        # (B, C, H, W) -> spec_logits: (B, num_spec, H, W)
        spec_logits = self.spec_router(x)
        
        # (B, C, H, W) -> shared_logits: (B, num_shared)
        shared_logits = self.shared_router(x)
        # Reshape shared_logits for spatial broadcasting: (B, num_shared, 1, 1)
        shared_logits = shared_logits.view(shared_logits.size(0), shared_logits.size(1), 1, 1)
        
        return self.core(x, spec_logits, shared_logits, self.spec_experts, self.shared_experts)

# ============================================================
# FULL MODEL: Adapted_SSMoE_UNet
# Replaces VanillaUNet bottleneck with SSMoE_ConvBlock
# ============================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Adapted_SSMoE_UNet(nn.Module):
    def __init__(self, in_channels=16, n_classes=1):
        super().__init__()
        # Encoder
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        # SWAP: Vanilla Down3 -> SSMoE_ConvBlock (Bottleneck)
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2), 
            SSMoE_ConvBlock(256, 512, num_spec=4, num_shared=2, top_k=2)
        )
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)
        
        # Output Head
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4)
        x = self.conv_up1(torch.cat([x, x3], dim=1))
        x = self.up2(x)
        x = self.conv_up2(torch.cat([x, x2], dim=1))
        x = self.up3(x)
        x = self.conv_up3(torch.cat([x, x1], dim=1))
        
        return torch.sigmoid(self.outc(x))

# ── Smoke Test ──
if __name__ == "__main__":
    print("[Validator] Verifying Adapted SSMoE Model shapes...")
    IN_CHANNELS = 16
    BATCH_SIZE = 2
    model = Adapted_SSMoE_UNet(in_channels=IN_CHANNELS, n_classes=1)
    
    dummy_input = torch.randn(BATCH_SIZE, IN_CHANNELS, 256, 256)
    try:
        output = model(dummy_input)
        print(f"✅ Success: Input {dummy_input.shape} -> Output {output.shape}")
        assert output.shape == (BATCH_SIZE, 1, 256, 256)
        print("✅ Tensor sanity check passed.")
    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
