# EEGMoE Deconstruction Report: Geospatial Adaptation for Landslide Detection

## 1. The Core Novelty
**Novelty Name:** Specific and Shared Mixture-of-Experts (SSMoE) Block
**Description:** Decouples domain-specific and domain-shared representations within a single encoder. It utilizes dual routing: Top-K sparse routing for "Specific Expert Groups" (task/region specialization) and Soft routing for "Shared Expert Groups" (common feature extraction).

---

## 2. Architectural Flow & Math
- **Original Domain:** 1D Signal Processing (EEG Transformers)
- **Target Domain:** 2D Geospatial Semantic Segmentation (Multi-channel Satellite Rasters)
- **Mathematical Flow:**
  1. **Input Tokens ($x$):** In 2D, $x$ represents either a pixel-wise feature vector or a flattened spatial patch.
  2. **Specific Router ($W_e$):** $g_x = W_e \cdot x$ (Logic: Compute routing scores for experts).
  3. **Specific MoE (Top-K):** $SpecMoE(x) = \sum_{i \in TopK} p_i(x)e_i(x)$ where $p_i(x) = \text{Softmax}(g_x)$.
  4. **Shared MoE (Fixed):** $ShareMoE(x) = \sum_{i \in F} p_i(x)f_i(x)$ where $F$ includes all shared experts.
  5. **Fusion (Residual Add):** $Output = SpecMoE(x) + ShareMoE(x)$.

---

## 3. Translation Blueprint (Hand-off to Cross-Domain Adapter)

| Original Component (EEG) | Required Translation (Geospatial 2D) |
| :--- | :--- |
| **1D Token $[S, D]$** | **2D Spatial Map $[C, H, W]$** (Treat each $[H, W]$ pixel as a token). |
| **Linear Expert ($e_i, f_i$)** | **$3 \times 3$ Conv2d / ResNet Block**. Each expert is a parallel spatial feature extractor. |
| **Linear Router ($W_e$)** | **$1 \times 1$ Conv2d**. Outputs a $K$-channel routing map, allowing spatially-variant expert selection. |
| **Soft Routing (Global)** | **Global Average Pooling + MLP**. Used if the "domain" selection is per-patch rather than per-pixel. |
| **$L_1$ Reconstruction Loss** | **BCE + Dice Loss** (Standard for Segmentation) or **MSE** (if using self-supervised reconstruction). |

> [!IMPORTANT]
> For Landslide Detection, the "Specific Experts" should be routed based on **Geographical Domain** (e.g., Terrain Type, Soil Moisture levels). The "Shared Experts" handle universal features like edge detection and spectral signatures of vegetation.

---

## 4. Training Constraints (Hand-off to Plumber/Validator)
- **Optimizer:** AdamW
- **Base Learning Rate:** 1e-4
- **Auxiliary Loss:** Load-balancing constraint ($L_{aux}$) is critical to prevent "Expert Collapse" where only one expert is used.
- **Masking:** For self-supervised pre-training, use **Spatial Patch Masking** (masking 40% of satellite pixels) rather than 1D token masking.

---

## 5. ⏱️ 4-Hour Adaptation Time Budget
- **T+00:15:** Initialize Baseline U-Net on Landslide Dataset.
- **T+01:00:** Implement `SSMoE_ConvBlock` with $1 \times 1$ Conv Routing and Parallel Experts.
- **T+02:00:** Pre-train on unlabeled satellite rasters using Patch Reconstruction Loss.
- **T+03:00:** Fine-tune on labeled Landslide Masks.
- **T+03:45:** A/B Analysis vs. Non-MoE Baseline.
