# EEG-Based P300 Brain Speller System
### Brain-Computer Interface (BCI) for Text Communication

This repository contains a complete end-to-end pipeline for an EEG-based Brain-Computer Interface. The system utilizes the **P300 Event-Related Potential (ERP)** to allow users to "type" characters by simply focusing on them as they flash on a 6×6 grid.

---

## 1. Project Overview

The core of this project is the detection of the **"Oddball" effect**. When a user focuses on a specific "Target" character among "Non-Target" characters, their brain produces a distinct positive voltage deflection roughly 300ms after the stimulus—the **P300**.

By processing noisy EEG signals, extracting spatial features, and applying machine learning, this system identifies the intended character with high precision.

### Key Technical Achievements
- **Peak Accuracy:** 92.4% (SVM-RBF)
- **Spatial Filtering:** Implemented **Xdawn** to isolate P300 components from background noise
- **Comparative Analysis:** Evaluated LDA, SVM, and Deep Learning (EEGNet) models
- **Metric Tracking:** Calculated Information Transfer Rate (ITR) to measure communication speed

---

## 2. System Architecture

The project is divided into four main stages:

1. **Preprocessing:** 0.1–30Hz bandpass filtering, 50Hz notch filtering, and ICA-based artifact removal
2. **Epoching:** Time-locking signal windows (-200ms to 800ms) relative to each flash
3. **Feature Extraction:** Downsampling and Xdawn spatial covariance mapping
4. **Classification:** Ensemble scoring across flash repetitions to determine the final character

---

## 3. Environment Setup

### Prerequisites
- **Python:** 3.9 or 3.10 (Recommended)
- **Virtual Environment:** Highly recommended to avoid dependency conflicts

### Installation

1. **Clone/Download** the repository to your local machine.

2. **Create a virtual environment:**
```bash
   python -m venv eeg_env
   eeg_env\Scripts\activate  # Windows
```

3. **Install Dependencies:**
```bash
   pip install mne numpy scipy scikit-learn matplotlib seaborn moabb pandas torch braindecode pyriemann autoreject
```

---

# 4. Project Structure

```
eeg_speller/
│
├── data/                    # EEG datasets (downloaded via MOABB)
│   └── .gitkeep
│
├── notebooks/               # Exploratory analysis and ERP visualization
│   └── erp_exploration.ipynb
│
├── src/                     # Core Python logic
│   ├── __init__.py          # Package marker
│   ├── preprocess.py        # Filtering and artifact removal
│   ├── features.py          # Xdawn and feature matrix creation
│   ├── models.py            # LDA, SVM, and EEGNet architectures
│   └── evaluate.py          # Cross-validation and ITR calculation
│
├── results/                 # Saved plots and performance logs
│   └── .gitkeep
│
└── README.md                # Project documentation
```
---

## 5. Instructions for Running the Code

### Phase 1: Local Execution (Data & Preprocessing)

1. Navigate to the project root folder.
2. Run the preprocessing script to download the dataset and clean the signals:
```bash
   python src/preprocess.py
```

### Phase 2: Cloud Execution (Google Colab)

For training the Deep Learning models (EEGNet), it is recommended to use GPU acceleration in Colab:

1. Upload the `eeg_speller` folder to your Google Drive.
2. Open a new Colab notebook and mount your drive:
```python
   from google.colab import drive
   drive.mount('/content/drive')

   import os
   os.chdir('/content/drive/MyDrive/eeg_speller')
```
3. Install libraries and execute the training script:
```python
   !pip install mne moabb braindecode
   !python src/models.py
```

---

## 6. Results Summary

| Method     | Accuracy | Precision | ITR (bits/min) |
|------------|----------|-----------|----------------|
| LDA        | 91.3%    | 92.1%     | High           |
| SVM (RBF)  | **92.4%**| **93.5%** | **Optimal**    |
| EEGNet     | 87.5%    | 89.2%     | Moderate       |

The SVM model with Xdawn spatial filtering proved the most reliable for this dataset, providing the best balance of speed and accuracy.

---

## 7. Author & Acknowledgments

- **Institution:** Indian Institute of Technology (IIT) Kanpur
- **Dataset:** BNCI2014_009 via the [MOABB](https://github.com/NeuroTechX/moabb) library
- **Special Thanks:** To my project mentor Manan Jindal for guidance on signal processing and BCI paradigms
