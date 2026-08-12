# Automatic Modulation Classification Engine (1D CNN)

A PyTorch pipeline that classifies radio signal modulation types directly from raw I/Q samples, using a 1D convolutional neural network trained on the RadioML 2016.10a dataset.

## Overview

Automatic Modulation Classification (AMC) is a core building block for cognitive radio and spectrum-sensing systems: before a receiver can demodulate a signal, it first needs to identify *how* that signal was modulated. This project trains a 1D CNN to make that call end-to-end from raw I/Q waveforms, with no hand-engineered features.

**Dataset:** [RadioML 2016.10a](https://www.deepsig.ai/datasets) — 220,000 labeled signal snippets across 11 modulation classes and 20 SNR levels (-20 dB to 18 dB), each snippet a `(2, 128)` I/Q sample.

**Modulation classes:** 8PSK, AM-DSB, AM-SSB, BPSK, CPFSK, GFSK, PAM4, QAM16, QAM64, QPSK, WBFM

## Pipeline

1. **Load & flatten** — unpack the RadioML pickle into `(X, y, SNR)` arrays
2. **Energy normalization** — scale each I/Q sample to unit average power
3. **Stratified split** — 70/15/15 train/val/test, stratified jointly on class + SNR so every split has balanced coverage across both
4. **Model** — a 3-block 1D CNN (Conv1d → BatchNorm → MaxPool, channels 64→128→256) feeding into fully connected classification layers
5. **Training** — Adam optimizer, `ReduceLROnPlateau` scheduling, early stopping on validation loss
6. **Evaluation** — overall test accuracy, accuracy vs. SNR curve, and a confusion matrix across all 11 classes

## Model Architecture

```
Input (2, 128)  →  I/Q raw samples
Conv1d(2→64, k=7)   + BatchNorm + MaxPool
Conv1d(64→128, k=5) + BatchNorm + MaxPool
Conv1d(128→256, k=3) + BatchNorm + MaxPool
Fully Connected → 11-class softmax
```

~373K trainable parameters.

## Results

| Metric | Value |
|---|---|
| Test accuracy (overall) | 61.1% |
| Peak accuracy (high SNR, ≥4 dB) | ~85–86% |
| Accuracy at 0 dB SNR | 82.6% |
| Accuracy at -10 dB SNR | 37.2% |

Accuracy climbs sharply from -20 dB up through 0 dB and then plateaus around 85% at higher SNRs — consistent with the expected behavior of AMC models on this dataset, where classification is fundamentally bounded by symbol overlap between similar-order modulations (e.g. QAM16 vs QAM64) even at high SNR.

See the notebook for the full accuracy-vs-SNR curve and confusion matrix.

## Requirements

```
torch
numpy
scikit-learn
matplotlib
seaborn
tqdm
```

## Usage

1. Download `RML2016.10a_dict.dat` from the RadioML dataset and place it in the project root (or update `DATA_PATH` in the notebook).
2. Open `CNN.ipynb` and run all cells top to bottom.
3. The best model checkpoint is saved to `best_amc_cnn1d.pth` during training and reloaded for final test evaluation.

## Project Structure

```
├── CNN.ipynb              # Full pipeline: data loading → training → evaluation
├── best_amc_cnn1d.pth     # Saved best model weights (generated on run)
└── README.md
```

## Author

R. M. S. H. Ratnayake — Department of Electronic and Telecommunication Engineering, University of Moratuwa

## License

MIT
