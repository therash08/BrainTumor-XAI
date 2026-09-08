# TrustSeg-XAI  
## Trustworthy Multi-Task Brain Tumor MRI Classification, Segmentation and Explainability

TrustSeg-XAI is a deep-learning research pipeline for **brain tumor MRI classification, segmentation, explainability, and causal faithfulness analysis**.

The central research question is:

> **Can a brain-tumor model make the correct diagnosis for the correct anatomical reason?**

Instead of evaluating only classification accuracy, this project investigates multiple dimensions of model trustworthiness:

- Brain tumor classification
- Tumor segmentation
- Explicit segmentation-guidance dependency
- Grad-CAM++ anatomical localization
- Causal tumor-occlusion faithfulness
- No-tumor safety
- Probability calibration
- Statistical significance and uncertainty

---

# 1. Project Overview

The study evaluates three classification models and three segmentation models.

## Classification Models

- EfficientNet-B0
- ConvNeXt-Tiny
- TrustSeg-XAI

## Segmentation Models

- U-Net with ResNet34 encoder
- U-Net++ with ResNet34 encoder
- TrustSeg-XAI segmentation branch

## Proposed Model

**TrustSeg-XAI** combines:

- ConvNeXt-Tiny shared image encoder
- U-Net-style segmentation decoder
- Global classification features
- Segmentation-guided ROI features
- Joint segmentation and classification learning

The model is designed to investigate whether segmentation-derived anatomical information contributes meaningfully to classification decisions.

---

# 2. Dataset

Dataset:

**Brain Tumor MRI Dataset**

Kaggle dataset identifier:

```text
maulikgajera/brain-tumor-mri-dataset
```

Project dataset root:

```text
D:\brain_tumor\🧠 Brain Tumor MRI Dataset
```

The dataset contains two related tasks:

```text
classification_task/
segmentation_task/
```

---

# 3. Classification Classes

Four MRI diagnostic classes are used:

| Class | Label |
|---|---:|
| Glioma | 0 |
| Meningioma | 1 |
| Pituitary | 2 |
| No Tumor | 3 |

MRI planes:

- Axial
- Coronal
- Sagittal

---

# 4. Dataset Size

## Classification

| Split | Images |
|---|---:|
| Original Training | 5,000 |
| Official Test | 1,000 |
| Total | 6,000 |

Official test distribution:

| Class | Images |
|---|---:|
| Glioma | 254 |
| Meningioma | 306 |
| Pituitary | 300 |
| No Tumor | 140 |
| **Total** | **1,000** |

## Expert Segmentation Masks

| Split | Expert Image-Mask Pairs |
|---|---:|
| Original Train | 3,933 |
| Official Test | 860 |
| Total | 4,793 |

No-tumor images do not contain expert tumor masks.

---

# 5. Data Audit and Leakage Control

A dedicated dataset audit was performed before model training.

The raw dataset was never modified.

Detected during audit:

```text
Exact duplicate SHA groups          : 46
Train-test exact overlap groups     : 7
Conflicting-label groups            : 0
```

Training images with exact test-set overlap were excluded using metadata.

Internal exact duplicate copies in the development set were also reduced to one representative.

Final exact-clean development set:

```text
4,955 MRI
```

After cleaning:

```text
Development ↔ Official Test SHA overlap = 0
Internal development exact duplicates   = 0
```

Perceptual-hash matches were treated only as **screening candidates**, not as confirmed leakage, and were not automatically deleted.

---

# 6. Final Development Split

The cleaned development dataset was split using seed `42`.

```text
Training       : 4,211
Validation     :   744
Official Test  : 1,000
```

The split was jointly stratified by:

```text
class + MRI plane
```

Expert-mask data after leakage cleaning:

```text
Expert Train       : 3,312
Expert Validation  :   585
Expert Test        :   860
```

The official test set was never used for model selection or hyperparameter tuning.

---

# 7. Preprocessing

Input resolution:

```text
256 × 256
```

Image processing:

- RGB conversion
- Bilinear image resizing
- ImageNet normalization
- Random horizontal flip during training
- Random rotation ±10°
- Identical geometric transformation for image and mask
- Nearest-neighbor resizing for masks
- Binary segmentation masks

ImageNet normalization:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

---

# 8. Local Development Environment

The study was executed locally.

## Hardware

```text
Operating System : Windows 11
GPU              : NVIDIA GeForce RTX 3050 Laptop GPU
VRAM             : 6 GB
CUDA             : 12.6
```

## Python

```text
Python 3.12.10
```

The CUDA-enabled PyTorch environment was used for training and evaluation.

Example installation:

```bash
pip install numpy pandas scipy scikit-learn matplotlib pillow tqdm
pip install timm segmentation-models-pytorch
```

CUDA PyTorch should be installed using the version appropriate for the local CUDA-compatible environment.

---

# 9. Reproducibility

Main random seed:

```text
42
```

Reproducibility settings included:

```python
random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

cuDNN settings:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

# 10. Baseline Segmentation Models

## U-Net ResNet34

Architecture:

```text
U-Net
Encoder: ResNet34
Encoder initialization: ImageNet pretrained
```

Training loss:

```text
0.5 × BCE
+
0.5 × Soft Dice Loss
```

Optimizer:

```text
AdamW
Learning rate = 1e-4
Weight decay  = 1e-4
```

Model selection:

```text
Best validation Dice
```

---

## U-Net++ ResNet34

Architecture:

```text
U-Net++
Encoder: ResNet34
Encoder initialization: ImageNet pretrained
```

The same training and evaluation protocol was used for fair comparison.

---

# 11. Baseline Classification Models

## EfficientNet-B0

Training objective:

```text
Cross-Entropy Loss
```

Optimizer:

```text
AdamW
Learning rate = 1e-4
Weight decay  = 1e-4
```

Selection metric:

```text
Validation Macro-F1
```

---

## ConvNeXt-Tiny

ConvNeXt-Tiny was trained using the same classification protocol and development split.

---

# 12. TrustSeg-XAI Architecture

TrustSeg-XAI uses a shared ConvNeXt-Tiny encoder.

```python
timm.create_model(
    "convnext_tiny",
    pretrained=True,
    features_only=True,
    out_indices=(0, 1, 2, 3)
)
```

The architecture contains three main components.

## 12.1 Shared Encoder

ConvNeXt-Tiny extracts hierarchical MRI features.

## 12.2 Segmentation Decoder

The decoder progressively reconstructs the tumor segmentation mask using skip-connected encoder features.

Conceptually:

```text
Deepest Encoder Feature
        ↓
Decoder + Skip Connections
        ↓
Tumor Segmentation Logits
        ↓
Predicted Tumor Guidance
```

## 12.3 ROI-Guided Classification

The deepest encoder representation is processed through two pathways.

### Global pathway

```text
Deep feature
→ Global Average Pooling
```

### ROI pathway

```text
Deep feature
×
Tumor guidance
→ ROI Average Pooling
```

The two feature representations are concatenated:

```text
Global feature + ROI feature
        ↓
LayerNorm
        ↓
Linear 512
        ↓
GELU
        ↓
Dropout 0.3
        ↓
4-class classifier
```

---

# 13. TrustSeg-XAI Multi-Task Objective

Total training loss:

```text
Ltotal =
0.5 × Lseg
+
0.5 × Lclassification
```

Segmentation loss:

```text
Lseg =
0.5 × BCE
+
0.5 × Soft Dice Loss
```

Classification loss:

```text
Cross-Entropy
```

Training configuration:

```text
Optimizer      : AdamW
Learning rate  : 1e-4
Weight decay   : 1e-4
Maximum epochs : 15
Early stopping : patience 3
Gradient clip  : 1
AMP            : enabled
```

Model selection score:

```text
0.5 × Validation Macro-F1
+
0.5 × Validation Expert Tumor Dice
```

---

# 14. Final Classification Results

Official test set:

```text
N = 1,000
```

| Model | Accuracy | Macro-F1 | MCC |
|---|---:|---:|---:|
| **EfficientNet-B0** | **0.996000** | **0.996544** | **0.994555** |
| TrustSeg-XAI | 0.990000 | 0.990557 | 0.986356 |
| ConvNeXt-Tiny | 0.986000 | 0.987241 | 0.980927 |

Classification accuracy in percentage form:

```text
EfficientNet-B0 : 99.60%
TrustSeg-XAI    : 99.00%
ConvNeXt-Tiny   : 98.60%
```

EfficientNet-B0 achieved the highest pure classification performance.

TrustSeg-XAI was not designed solely to maximize classification accuracy; its main contribution is the integrated evaluation of anatomical and causal trustworthiness.

---

# 15. Classification Bootstrap Confidence Intervals

10,000 image-level bootstrap resamples were used.

## EfficientNet-B0

```text
Accuracy:
0.996000 [0.992000, 0.999000]

Macro-F1:
0.996544 [0.992900, 0.999197]
```

## ConvNeXt-Tiny

```text
Accuracy:
0.986000 [0.978000, 0.993000]

Macro-F1:
0.987241 [0.979987, 0.993503]
```

## TrustSeg-XAI

```text
Accuracy:
0.990000 [0.984000, 0.996000]

Macro-F1:
0.990557 [0.984357, 0.996082]
```

---

# 16. Final Segmentation Results

Expert tumor test cohort:

```text
N = 860
```

| Model | Dice | IoU | Precision | Recall | Specificity | HD95 |
|---|---:|---:|---:|---:|---:|---:|
| U-Net ResNet34 | 0.832102 | 0.740506 | 0.840236 | 0.849692 | 0.997373 | 8.482558 |
| U-Net++ ResNet34 | 0.835310 | 0.743632 | 0.856876 | 0.839939 | 0.997560 | 9.367526 |
| **TrustSeg-XAI** | **0.839223** | **0.747783** | **0.865963** | **0.840963** | **0.997870** | **7.907168** |

TrustSeg-XAI achieved the highest local point estimate for:

- Dice
- IoU
- Precision
- Specificity

and the lowest standard HD95 among the three evaluated segmentation models.

TrustSeg-XAI Dice bootstrap 95% CI:

```text
0.839223
[0.828031, 0.849536]
```

---

# 17. Explicit Segmentation-Guidance Dependency

Four classification guidance conditions were tested:

1. Predicted segmentation guidance
2. Expert target guidance
3. No guidance mask
4. Spatially corrupted guidance mask

Official test:

```text
N = 1,000
```

Results:

| Condition | Accuracy | Macro-F1 |
|---|---:|---:|
| Predicted | 0.990000 | 0.990557 |
| Target | 0.990000 | 0.990557 |
| No Mask | 0.990000 | 0.990557 |
| Corrupted Mask | 0.990000 | 0.990557 |

Hard prediction changes:

```text
Predicted vs Target    : 0 / 1000
Predicted vs No Mask   : 0 / 1000
Predicted vs Corrupted : 0 / 1000
```

Mean true-class probabilities:

```text
Predicted : 0.989740
Target    : 0.989769
No Mask   : 0.989379
Corrupted : 0.989371
```

Interpretation:

> TrustSeg-XAI showed negligible hard-decision dependence on the explicit segmentation-guidance pathway. Small probability changes were detectable, but their practical magnitude was extremely small.

This indicates that the global image pathway dominates the final classification decision.

---

# 18. Grad-CAM++ Anatomical Localization

Grad-CAM++ was evaluated using a permanently fixed subset of:

```text
150 tumor-positive MRI
```

Sampling was balanced:

```text
50 glioma
50 meningioma
50 pituitary

50 axial
50 coronal
50 sagittal
```

Correct classifications among the subset:

```text
148 / 150
```

## Overall Grad-CAM++ Results

```text
XAI Dice                : 0.341751
XAI IoU                 : 0.236065
Pointing Game           : 0.373333
Tumor Saliency Fraction : 0.194962
Mean CAM Area Ratio     : 0.037159
Degenerate CAMs         : 17 / 150
```

Bootstrap 95% CI for XAI IoU:

```text
0.236065
[0.204420, 0.268462]
```

Interpretation:

> Grad-CAM++ showed partial but incomplete overlap with expert tumor anatomy.

Weak or incomplete CAM localization alone should not be interpreted as proof that the classifier ignores tumor information.

---

# 19. Grad-CAM++ Class-Wise Results

| Class | XAI Dice | XAI IoU | Pointing Game | Tumor Saliency Fraction |
|---|---:|---:|---:|---:|
| Glioma | 0.329886 | 0.223566 | 0.480000 | 0.220687 |
| Meningioma | 0.373949 | 0.282000 | 0.460000 | 0.252666 |
| Pituitary | 0.321418 | 0.202630 | 0.180000 | 0.111532 |

Pituitary MRI showed the weakest Pointing Game performance.

---

# 20. Grad-CAM++ Plane-Wise Results

| Plane | XAI Dice | XAI IoU | Pointing Game | Tumor Saliency Fraction |
|---|---:|---:|---:|---:|
| Axial | 0.310690 | 0.215675 | 0.320000 | 0.187557 |
| Coronal | 0.301566 | 0.203737 | 0.340000 | 0.183423 |
| Sagittal | **0.412997** | **0.288785** | **0.460000** | **0.213905** |

Sagittal MRI demonstrated the strongest Grad-CAM++ localization in this local experiment.

---

# 21. Causal Tumor-Occlusion Faithfulness

The exact same locked 150-image subset used for Grad-CAM++ was reused for causal faithfulness analysis.

Each MRI was evaluated under:

1. Original MRI
2. Expert-tumor-occluded MRI
3. Area-matched background-occluded MRI

The target class was the permanently locked Phase-9 prediction.

Background controls preserved the tumor-mask area.

```text
Translated exact-shape controls : 149
Equal-area fallback             : 1
```

---

# 22. Causal Faithfulness Results

```text
Mean ΔP Tumor       : 0.276489
Mean ΔP Background  : 0.000154
Mean Margin         : 0.276334

Median Margin       : ~0.000001

Tumor > Background  : 62.0%

Tumor Flip Rate     : 28.0%
Background Flip Rate: 0.0%
```

One-sided paired Wilcoxon test:

```text
p = 3.29624718469e-08
```

Bootstrap 95% CI for mean faithfulness margin:

```text
0.276334
[0.205139, 0.349919]
```

Interpretation:

> Removal of expert-defined tumor information produced a substantially greater average effect on the model decision than matched-background removal.

However, the near-zero median margin indicates substantial case-level heterogeneity.

Therefore:

> The model demonstrates strong **average** causal tumor dependence, but this dependence is not uniformly strong across every MRI.

---

# 23. Causal Faithfulness by Tumor Class

| Class | Tumor ΔP | Background ΔP | Mean Margin | Tumor Flip Rate |
|---|---:|---:|---:|---:|
| Glioma | **0.616182** | 0.000000 | **0.616181** | **0.620000** |
| Meningioma | 0.154616 | 0.000450 | 0.154166 | 0.160000 |
| Pituitary | 0.058669 | 0.000014 | 0.058655 | 0.060000 |

Glioma predictions showed substantially stronger causal tumor dependence than the other classes.

---

# 24. Causal Faithfulness by MRI Plane

| Plane | Tumor ΔP | Background ΔP | Mean Margin | Tumor Flip Rate |
|---|---:|---:|---:|---:|
| Axial | 0.295572 | 0.000007 | 0.295564 | 0.300000 |
| Coronal | 0.273945 | 0.000456 | 0.273489 | 0.280000 |
| Sagittal | 0.259949 | ~0.000000 | 0.259949 | 0.260000 |

---

# 25. Why Mask Dependency and Causal Faithfulness Are Not Contradictory

The mask-dependency experiment changed only the explicit ROI guidance mask.

The original MRI still contained the tumor.

Therefore, the global ConvNeXt pathway retained access to tumor information.

In contrast, causal occlusion physically removed expert-defined tumor pixels from the input MRI.

Therefore:

> Low dependence on the explicit ROI mask does **not** imply low dependence on tumor image content.

The findings indicate:

```text
Explicit ROI-guidance dependency:
Very weak

Actual tumor-content dependency:
Strong on average
```

---

# 26. No-Tumor Safety

Official no-tumor test cohort:

```text
N = 140
```

## Classification Safety

```text
Accuracy                  : 1.000000
Mean no-tumor probability : 0.999996
Mean tumor probability    : 0.000004
```

## Segmentation Safety

Using the permanently fixed threshold:

```text
Threshold = 0.5
```

Results:

```text
Mean false tumor area    : 0.000000
Median false tumor area  : 0.000000
Maximum false tumor area : 0.000000

Empty prediction rate    : 1.000000
False tumor detection    : 0.000000

Mean maximum segmentation probability:
0.025482
```

All 140 no-tumor MRI produced empty binary segmentation masks.

Interpretation:

> TrustSeg-XAI demonstrated excellent no-tumor safety on the evaluated official test cohort.

This result should not be interpreted as proof of universal clinical safety.

---

# 27. Calibration Analysis

Calibration was evaluated on the exact same 1,000 official test MRI.

Metrics:

- 15-bin Expected Calibration Error
- Multiclass Brier Score

No:

- probability renormalization
- temperature scaling
- post-hoc test-set calibration

was performed.

## Final Calibration Results

| Model | Accuracy | Mean Confidence | ECE | Brier |
|---|---:|---:|---:|---:|
| **EfficientNet-B0** | **0.996000** | 0.997679 | **0.003536** | **0.007649** |
| TrustSeg-XAI | 0.990000 | 0.999375 | 0.009990 | 0.019438 |
| ConvNeXt-Tiny | 0.986000 | 0.996055 | 0.011382 | 0.024853 |

Lower ECE and Brier scores indicate better calibration.

Therefore:

```text
Best calibrated classifier:
EfficientNet-B0
```

All three models were slightly overconfident on average.

---

# 28. Statistical Classifier Comparison

Exact paired McNemar tests were performed on the same 1,000 MRI.

## EfficientNet-B0 vs ConvNeXt-Tiny

```text
EfficientNet correct / ConvNeXt wrong = 12
EfficientNet wrong / ConvNeXt correct = 2

Accuracy difference = +0.010000

p = 0.012939453125
```

Result:

```text
Statistically significant at α = 0.05
```

---

## EfficientNet-B0 vs TrustSeg-XAI

```text
EfficientNet correct / TrustSeg wrong = 9
EfficientNet wrong / TrustSeg correct = 3

Accuracy difference = +0.006000

p = 0.14599609375
```

Result:

```text
Not statistically significant
```

---

## ConvNeXt-Tiny vs TrustSeg-XAI

```text
ConvNeXt correct / TrustSeg wrong = 6
ConvNeXt wrong / TrustSeg correct = 10

Accuracy difference = -0.004000

p = 0.454498291016
```

Result:

```text
Not statistically significant
```

---

# 29. Integrated Scientific Finding

The most important conclusion of this study is:

> **High classification performance does not automatically imply strong anatomical localization.**

TrustSeg-XAI achieved strong diagnostic performance and strong segmentation performance, yet Grad-CAM++ only partially overlapped expert tumor anatomy.

At the same time, direct tumor occlusion showed a substantial average reduction in class probability compared with matched-background removal.

Therefore:

> **Post-hoc saliency localization and causal faithfulness measure different dimensions of model trustworthiness.**

A model may depend causally on tumor information even when its Grad-CAM++ heatmap does not closely reproduce the expert segmentation mask.

---

# 30. Project Folder Structure

Recommended cleaned project structure:

```text
brain_tumor/
│
├── 🧠 Brain Tumor MRI Dataset/
│   ├── classification_task/
│   └── segmentation_task/
│
├── metadata/
│
├── unet_resnet34/
│   ├── checkpoints/
│   └── results/
│
├── unetplusplus_resnet34/
│   ├── checkpoints/
│   └── results/
│
├── efficientnet_b0/
│   ├── checkpoints/
│   └── results/
│
├── convnext_tiny/
│   ├── checkpoints/
│   └── results/
│
├── trustseg_xai/
│   ├── checkpoints/
│   └── results/
│
├── mask_dependency/
│   └── results/
│
├── gradcampp/
│   └── results/
│
├── causal_faithfulness/
│   └── results/
│
├── no_tumor_safety/
│   └── results/
│
├── calibration/
│   └── results/
│
├── statistical_analysis/
│   └── results/
│
├── final_outputs/
│   │
│   ├── tables/
│   │   ├── table1_classification_performance.csv
│   │   ├── table2_segmentation_performance.csv
│   │   ├── table3a_mask_dependency_conditions.csv
│   │   ├── table3b_mask_dependency_statistics.csv
│   │   ├── table4a_gradcam_overall.csv
│   │   ├── table4b_gradcam_classwise.csv
│   │   ├── table4c_gradcam_planewise.csv
│   │   ├── table5a_causal_overall.csv
│   │   ├── table5b_causal_classwise.csv
│   │   ├── table5c_causal_planewise.csv
│   │   ├── table6a_no_tumor_safety.csv
│   │   ├── table6b_no_tumor_planewise.csv
│   │   ├── table7_calibration.csv
│   │   ├── table8_mcnemar.csv
│   │   └── table9_bootstrap_ci.csv
│   │
│   ├── figures/
│   │   ├── figure1_classification_performance.png
│   │   ├── figure2_segmentation_dice.png
│   │   ├── figure3_calibration_ece.png
│   │   ├── figure4_reliability_*.png
│   │   ├── figure5_mask_dependency_probability_effect.png
│   │   ├── figure6_gradcam_classwise_dice.png
│   │   └── figure7_causal_faithfulness_classwise.png
│   │
│   └── summary/
│       ├── paper_ready_results_summary.md
│       ├── trustseg_xai_final_master_results.json
│       └── figure_manifest.csv
│
├── notebook.ipynb
│
└── README.md
```

---

# 31. Experimental Workflow

The complete study follows this sequence:

```text
Dataset
  ↓
Dataset Structure Verification
  ↓
Leakage / Duplicate Audit
  ↓
Clean Development Split
  ↓
Preprocessing + DataLoaders
  ↓
U-Net ResNet34
  ↓
U-Net++ ResNet34
  ↓
EfficientNet-B0
  ↓
ConvNeXt-Tiny
  ↓
TrustSeg-XAI Training
  ↓
Locked Official-Test Evaluation
  ↓
Segmentation-Guidance Dependency
  ↓
Grad-CAM++ Localization
  ↓
Causal Tumor Occlusion
  ↓
No-Tumor Safety
  ↓
Calibration
  ↓
McNemar + Bootstrap Statistics
  ↓
Publication Tables and Figures
```

---

# 32. Running the Project

The primary experimental workflow is contained in:

```text
notebook.ipynb
```

Run the notebook sequentially.

Do not use official test results to alter:

- model architecture
- hyperparameters
- segmentation threshold
- Grad-CAM threshold
- model selection
- training duration

The official test set is reserved for final locked evaluation.

---

# 33. Final Publication Outputs

All final paper-ready outputs are stored in:

```text
final_outputs/
```

## Tables

```text
final_outputs/tables/
```

## Figures

```text
final_outputs/figures/
```

## Study Summaries

```text
final_outputs/summary/
```

The consolidated machine-readable results are stored in:

```text
trustseg_xai_final_master_results.json
```

---

# 34. Recommended Primary Metrics for Reporting

## Classification

Report:

```text
Accuracy
Macro-F1
MCC
ROC-AUC
Bootstrap 95% CI
```

## Segmentation

Report:

```text
Dice
IoU
Precision
Recall
Specificity
HD95
```

## Explainability

Report:

```text
XAI Dice
XAI IoU
Pointing Game
Tumor Saliency Fraction
```

## Causal Faithfulness

Report:

```text
ΔP Tumor
ΔP Background
Faithfulness Margin
Tumor > Background proportion
Decision Flip Rates
Wilcoxon p-value
Bootstrap 95% CI
```

## Calibration

Report:

```text
ECE
Multiclass Brier Score
```

---

# 35. Important Interpretation Rules

The following interpretations should remain fixed when reporting this study.

### 1. EfficientNet-B0 is the best pure classifier

It achieved the highest classification accuracy and best calibration.

### 2. TrustSeg-XAI is not claimed to be the best classifier

Its primary contribution is the integrated trustworthiness analysis.

### 3. TrustSeg-XAI achieved the highest local segmentation Dice point estimate

However, pairwise statistical superiority over the segmentation baselines was not tested.

### 4. Weak explicit ROI dependency does not imply weak tumor dependency

The global pathway still sees the original MRI.

### 5. Grad-CAM++ localization does not equal causal faithfulness

These are separate forms of evidence.

### 6. Statistical significance does not automatically imply practical importance

For example, tiny guidance-mask probability shifts may be statistically detectable while remaining practically negligible.

---

# 36. Limitations

Important limitations include:

1. The study is based on a single public brain MRI dataset.

2. External hospital-domain validation was not performed.

3. The fixed local 150-image Grad-CAM++ subset is deterministic but should not be claimed to contain the exact same MRI identities as any earlier experiment unless explicitly verified.

4. Grad-CAM++ is a post-hoc attribution method and should not be interpreted as direct mechanistic evidence.

5. Causal occlusion is stronger evidence than saliency overlap but still represents an intervention on image pixels rather than a prospective clinical experiment.

6. The causal background-control procedure used translated area- and shape-matched controls when possible, with one deterministic equal-area fallback.

7. TrustSeg-XAI showed heterogeneous causal dependence across individual MRI and tumor classes.

8. High test-set performance does not establish clinical deployment readiness.

9. No-tumor safety results apply specifically to the evaluated 140-image official no-tumor cohort.

10. Additional external, multi-center, scanner-diverse and clinically curated validation is required before considering clinical use.

---

# 37. Reproducibility Policy

This repository follows several safeguards:

```text
Raw dataset modification          : No
Exact train-test leakage retained : No
Official test used for tuning     : No
Test-driven retraining            : No
Calibration fitting on test       : No
Threshold optimization on test    : No
```

Model checkpoints were locked before downstream trustworthiness analyses.

---

# 38. Key Final Results

```text
CLASSIFICATION
----------------------------------------
EfficientNet-B0 Accuracy : 99.60%
TrustSeg-XAI Accuracy    : 99.00%
ConvNeXt-Tiny Accuracy   : 98.60%


SEGMENTATION
----------------------------------------
TrustSeg-XAI Dice        : 83.92%
U-Net++ Dice             : 83.53%
U-Net Dice               : 83.21%


MASK DEPENDENCY
----------------------------------------
Hard prediction changes  : 0 / 1000


GRAD-CAM++
----------------------------------------
XAI Dice                 : 0.341751
XAI IoU                  : 0.236065
Pointing Game            : 0.373333


CAUSAL FAITHFULNESS
----------------------------------------
Tumor ΔP                 : 0.276489
Background ΔP            : 0.000154
Faithfulness Margin      : 0.276334
Wilcoxon p               : 3.30e-08


NO-TUMOR SAFETY
----------------------------------------
Classification Accuracy  : 100%
False Tumor Detection    : 0%


CALIBRATION
----------------------------------------
Best Model               : EfficientNet-B0
ECE                      : 0.003536
Brier                    : 0.007649
```

---

# 39. Main Research Conclusion

The study demonstrates that evaluating only diagnostic accuracy is insufficient for trustworthy medical AI.

TrustSeg-XAI showed:

- high classification performance,
- strong tumor segmentation,
- negligible hard dependence on explicit ROI guidance,
- partial Grad-CAM++ anatomical localization,
- substantial average causal dependence on tumor image content,
- excellent no-tumor behavior on the evaluated cohort,
- and measurable class-specific heterogeneity.

The central finding is:

> **A model can rely causally on tumor information even when post-hoc saliency maps only partially overlap the expert-defined tumor region.**

Therefore, trustworthy medical-AI evaluation should combine:

```text
Predictive Performance
+
Segmentation Quality
+
Post-hoc Localization
+
Causal Faithfulness
+
Negative-Control Safety
+
Calibration
+
Statistical Uncertainty
```

rather than relying on a single explainability metric.

---

# 40. Research / Educational Use

This project is intended for:

- Academic research
- Medical-AI experimentation
- Explainable-AI research
- Deep-learning education
- Reproducibility studies

It is **not intended for direct clinical diagnosis or treatment decisions**.

---

# 41. Citation

If this work becomes part of a publication, thesis, conference paper, or public repository, replace the following placeholder with the final citation.

```bibtex
@article{trustsegxai2026,
  title   = {TrustSeg-XAI: Trustworthy Brain Tumor MRI Classification,
             Segmentation and Causal Explainability},
  author  = {Author Name},
  year    = {2026},
  note    = {Research implementation}
}
```

---

# 42. Author

```text
Author      : [Rasidul Hoque Chowdhury]
Project     : TrustSeg-XAI
Domain      : Medical Image Analysis / Deep Learning / Explainable AI
Year        : 2026
```

---

## Status

```text
Dataset Audit              : Complete
Baseline Segmentation      : Complete
Baseline Classification    : Complete
TrustSeg-XAI Training      : Complete
Official Test Evaluation   : Complete
Mask Dependency            : Complete
Grad-CAM++                 : Complete
Causal Faithfulness        : Complete
No-Tumor Safety            : Complete
Calibration                : Complete
Statistical Analysis       : Complete
Publication Outputs        : Complete
```

# TRUSTSEG-XAI EXPERIMENTAL PIPELINE COMPLETE# BrainTumor-XAI
