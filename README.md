# CERTIFY-BTC

**A certification-oriented, multi-scale brain-tumor MRI classification research framework.**

CERTIFY-BTC is a 4-class brain tumor classifier (`glioma`, `meningioma`, `notumor`, `pituitary`)
built as a **research prototype** for an IEEE Transactions on Medical Imaging submission.
Its focus is not just accuracy but *trustworthiness*: uncertainty quantification,
out-of-distribution rejection, multi-site robustness, and explainability.

> ⚠️ **This is a research prototype, NOT a certified medical device.**
> It is **not** approved, cleared, or validated for clinical use. Nothing it outputs is
> "medical grade" or "certified for diagnosis." Do not use it to make patient decisions.

It is the successor to **HXAI-BTC** (EfficientNetB4 + CBAM, ~93% test accuracy, with
glioma as the weak class at ~82%). HXAI-BTC serves as the **baseline row** in our ablation study.

---

## Why "Certify"?

Most tumor classifiers report a single accuracy number. A model you could trust in a
research/clinical *pipeline* also has to answer: *how sure are you?*, *is this scan even
a brain MRI?*, *does this hold on data from a different hospital?*, and *why did you decide that?*
CERTIFY-BTC is designed around answering those four questions.

---

## Target architecture

Built in **verifiable phases** (see roadmap) — not all of this exists yet.

1. **Preprocessing** — skull-strip (HD-BET), N4 bias correction, CLAHE, resize 380×380
2. **Multi-scale backbone** — EfficientNetB4; extract Block 3 + Block 5 + Block 7, each with
   its own **CBAM** attention module, pooled + concatenated → FC(512)
3. **Two heads** —
   (a) tumor classifier using **Evidential Deep Learning** (Dirichlet output, single-pass uncertainty);
   (b) domain classifier with a **Gradient Reversal Layer** for domain-adversarial training across datasets
4. **CBAM pseudo-mask** — turn the CBAM spatial attention map into a segmentation mask
   (upsample → Gaussian blur → Otsu → morphological close/open → largest connected component)
5. **Radiomics fusion** — PyRadiomics (107 features) on the pseudo-masked region → PCA(32) →
   concatenated with deep features → final classifier
6. **Certification layer** — Evidential uncertainty + **Conformal Prediction** (RAPS, 95% coverage) +
   **OOD detection** (energy score)
7. **Explainability** — Grad-CAM (45%) + LIME (30%) + SHAP (25%) consensus heatmap; pseudo-mask
   validated against BraTS ground-truth via **Dice**; gradient-based counterfactuals

---

## Datasets

Multi-dataset training is the credibility core. The loader (Phase 1) handles per-dataset
class mappings and can combine them.

| Dataset | Classes | Role |
|---|---|---|
| Nickparvar (Kaggle) | 4 | Primary training |
| Figshare | 3 | Cross-dataset validation |
| BraTS | seg. masks | Training + pseudo-mask Dice validation |
| TCGA-GBM/LGG | — | Multi-site robustness |

**Datasets are never committed to this repo** (see `.gitignore`). Place them under `data/`
locally; patient imaging must not be redistributed.

---

## Two run modes — one switch

Everything that differs between hardware lives in [`config.py`](config.py) behind a single flag:

```python
MACHINE = "local"   # RTX 4050 6GB: batch 4, 2 epochs, Nickparvar only, heavy modules OFF (DEBUG)
MACHINE = "cloud"   # rented GPU: batch 32, full epochs, all datasets, everything ON
```

Every module must run end-to-end on the RTX 4050 in `local` mode before we scale to `cloud`.
Training uses **Automatic Mixed Precision (fp16)** and **gradient accumulation** to fit 6 GB,
and saves a checkpoint **every epoch** to `checkpoints/`.

Print the active config any time:

```bash
python config.py
```

---

## Repo structure

```
CERTIFY_BTC/
├── config.py              # single source of truth: MACHINE flag, paths, hyperparams
├── requirements.txt
├── .gitignore             # ignores data/, checkpoints/, __pycache__, *.pth
├── README.md
├── modules/
│   ├── preprocessing.py
│   ├── datasets.py        # flexible multi-dataset loader
│   ├── cbam.py            # CBAM (channel + spatial attention)
│   ├── model.py           # multi-scale EfficientNetB4 + CBAM + heads
│   ├── losses.py          # Focal Loss, Evidential loss
│   ├── domain_adversarial.py
│   ├── cbam_mask.py
│   ├── radiomics_fusion.py
│   ├── conformal.py
│   ├── ood.py
│   ├── uncertainty.py     # Evidential DL + MC Dropout
│   ├── xai.py             # Grad-CAM, LIME, SHAP, consensus
│   └── counterfactual.py
├── notebooks/
│   ├── 01_data_explore.ipynb
│   ├── 02_train.ipynb
│   └── 03_results_xai.ipynb
└── train.py               # main training entry point
```

---

## Setup

> Use **Python 3.11**, not 3.13. The radiomics / medical-imaging stack (pyradiomics,
> SimpleITK) has clean wheels on 3.11 but breaks on 3.13.

```powershell
# 1. Create the venv with Python 3.11 and activate it (Windows PowerShell).
#    Replace the path if your 3.11 lives elsewhere: find it with `py --list`.
& "C:\Users\VICTUS\AppData\Roaming\uv\python\cpython-3.11.15-windows-x86_64-none\python.exe" -m venv venv
.\venv\Scripts\Activate.ps1
python --version                      # should print 3.11.15

# 2. Install PyTorch matched to your CUDA (RTX 4050 driver supports CUDA 12.6)
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 3. Install the rest
pip install -r requirements.txt

# 4. Verify the GPU is visible
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 5. Print the active config
python config.py
```

---

## Build roadmap

The project is built one verifiable phase at a time; each must run on the RTX 4050 in
`local` mode before moving on.

- [x] **Phase 0** — repo scaffold: folders, `config.py`, `requirements.txt`, `.gitignore`, `README.md`
- [x] **Phase 1** — `datasets.py` + `preprocessing.py` (load Nickparvar; `01_data_explore.ipynb`)
- [x] **Phase 2** — `cbam.py` + `model.py` (multi-scale backbone; forward-pass shape check)
- [x] **Phase 3** — `losses.py` + `train.py` (Focal Loss + Stage-1 loop, AMP + checkpointing)
- [x] **Phase 4** — `cbam_mask.py` (attention→brain-confined pseudo-mask) + `radiomics_fusion.py` (skimage features→PCA32→fuse)
- [x] **Phase 5** — `domain_adversarial.py` (GRL + domain head) + Stage-2 loop (`train.py --stage 2`); mechanism verified locally, real multi-dataset on cloud
- [ ] **Phase 6** — `uncertainty.py` (EDL) + `conformal.py` + `ood.py`
- [ ] **Phase 7** — `xai.py` + `counterfactual.py`
- [ ] **Phase 8** — full results notebook + ablation study

---

## License & data ethics

Code license: TBD. Datasets retain their original licenses and are **not** included here.
This repository contains no patient data. Handle all MRI data in accordance with the
source datasets' terms and applicable privacy regulations.
