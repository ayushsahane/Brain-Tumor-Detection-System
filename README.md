<p align="center">
  <img src="https://raw.githubusercontent.com/Oct4Pie/brain-tumor-detection/main/logo/brain.png" width="12%" alt="Brain logo" />
</p>

# Brain Tumor Detection System (CNN)

A beginner‑friendly, end‑to‑end Streamlit app for brain tumor classification from 2D MRI scans using a Convolutional Neural Network (CNN).

<p align="center">
  <img src="https://i.imgur.com/C0rTivW.png" alt="App preview" />
</p>

---

## What this project does
- Takes a **2D MRI image** as input
- Crops the brain region automatically
- Runs a **CNN model** to classify **Tumor / No Tumor**
- Shows prediction confidence and evaluation metrics

---

## Quick Start (Beginner Friendly)

> **Recommended Python:** 3.10 or 3.11 (TensorFlow is most stable on these)

### 1) Clone the repository
```bash
git clone https://github.com/ayushsahane/Brain-Tumor-Detection-System.git
cd Brain-Tumor-Detection-System
```

### 2) Create & activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

> On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

### 4) Run the Streamlit app
```bash
python -m streamlit run app.py
```

---

## 🧠 Model Information

### Model Summary
```
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                      │ (None, 50, 50, 32)          │             896 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d (MaxPooling2D)         │ (None, 25, 25, 32)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_1 (Conv2D)                    │ (None, 25, 25, 64)          │          18,496 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_1 (MaxPooling2D)       │ (None, 12, 12, 64)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 12, 12, 64)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (Flatten)                    │ (None, 9216)                │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 128)                 │       1,179,776 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 1)                   │             129 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 3,597,893 (13.72 MB)
 Trainable params: 1,199,297 (4.57 MB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 2,398,596 (9.15 MB)
```

### Model Report
```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       209
           1       1.00      0.99      1.00       191

    accuracy                           1.00       400
   macro avg       1.00      1.00      1.00       400
weighted avg       1.00      1.00      1.00       400
```

---

##  How to Use the App
- Use **pre‑loaded samples** or upload your own MRI image
- The app crops the brain region automatically
- You receive a **Tumor / No Tumor prediction** with confidence

---

##  Project Structure
```
├── README.md
├── app.py                    # Streamlit entry point
├── logo
│   └── brain.png
├── model
│   ├── class_rep.py          # Classification report utilities
│   ├── mask.py               # Brain extraction + cropping
│   ├── modeler.py            # CNN training code
│   ├── plot.py               # Metric plotting
│   ├── predict.py            # Prediction helpers
│   └── predictor.py          # Model loading + evaluation
├── pages
│   ├── _pages
│   │   ├── about.py
│   │   ├── components.py
│   │   ├── github.py
│   │   ├── home.py
│   │   ├── try_it.py
│   │   └── utils.py
│   ├── components
│   │   ├── github_card.html
│   │   ├── github_iframe.html
│   │   └── title.html
│   ├── css
│   │   └── streamlit.css
│   └── samples               # Sample validation images
├── requirements.txt
└── temp.png
```

---

##  Beginner‑Friendly Glossary
- **CNN (Convolutional Neural Network):** A neural network specialized for images.
- **MRI (Magnetic Resonance Imaging):** Medical imaging technique for brain scans.
- **Threshold:** Value that decides Tumor vs No Tumor (0–1 probability).
- **Precision / Recall / F1:** Standard ML metrics to judge prediction quality.

---

##  Common Issues & Fixes

###  `ImportError: numpy.core.umath failed to import`
Use compatible NumPy:
```bash
pip install "numpy<2"
```

###  `ModuleNotFoundError: No module named 'cv2'`
Install OpenCV:
```bash
pip install opencv-python
```

###  Streamlit opens but errors show from Anaconda
Run using the virtual environment:
```bash
source venv/bin/activate
python -m streamlit run app.py
```

---

##  Acknowledgements
- [Brain Tumor Classification (MRI)](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri)
- [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)
- [MRI Based Brain Tumor Images](https://www.kaggle.com/mhantor/mri-based-brain-tumor-images)
- [Starter: Brain MRI Images for Brain](https://www.kaggle.com/kerneler/starter-brain-mri-images-for-brain-b5be8b94-c)

---

##  License
This project is for educational and research purposes. Use responsibly.
