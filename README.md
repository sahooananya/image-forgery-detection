# 🖼️ Image Forgery Detection using CNN

This project detects whether an image is **authentic** or **tampered** using a **Convolutional Neural Network (CNN)** trained on three public forgery datasets: CASIA2, Columbia, and COVERAGE.

---

## 🚀 Project Overview

Image forgery is the act of manipulating digital images without leaving visible traces. This project leverages deep learning to classify images into two categories:
- `authentic` (original images)
- `tampered` (forged or altered images)

The model was trained using a custom CNN architecture and evaluated using metrics like accuracy, confusion matrix, and classification report.

---

## 🧠 Model Performance

| Metric             | Value     |
|--------------------|-----------|
| ✅ Accuracy         | **73.1%** |
| 🎯 Precision (T)    | 0.65      |
| 🔁 Recall (T)       | 0.74      |
| 📊 F1-Score (T)     | 0.69      |

> `T = tampered class`

---

## 🗂️ Dataset Used

The dataset is compiled from:
- **CASIA2**
- **COVERAGE**
- **Columbia Uncompressed Image Splicing Dataset**

Images were preprocessed, resized to `128x128`, normalized, and split using an 80:20 ratio for training and testing.

---

## 🛠️ Project Structure

```bash
├── dataset/                   # Raw datasets (CASIA2, Columbia, etc.)
├── final_dataset/            # Train-test split organized into folders
│   ├── train/
│   └── test/
├── forgery_dataset.npz       # Numpy compressed file (preprocessed data)
├── prepare_final_dataset.py          # Script to process, merge, and save datasets
├── model_build_train.py      # Builds and trains the CNN model
├── model_evaluate.py         # Evaluates model performance
├── forgery_cnn_model.h5      # Saved baseline model (no augmentation)
├── forgery_cnn_augmented.h5  # Saved model with image augmentation
└── README.md                 # Project documentation
