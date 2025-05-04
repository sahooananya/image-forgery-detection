# 🔍 Image Forgery Detection using CNN

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
| ✅ Accuracy         | **74.87%** |
| 🎯 Precision (T)    | 0.67      |
| 🔁 Recall (T)       | 0.75      |
| 📊 F1-Score (T)     | 0.71      |

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
├── dataset/                        # Raw datasets (CASIA2, Columbia, etc.)
├── final_dataset/                 # Train-test split organized into folders
│   ├── train/
│   └── test/
├── forgery_dataset.npz            # Numpy compressed file (preprocessed data)
├── prepare_final_dataset.py       # Script to process, merge, and save datasets
├── model_build_train.py           # Builds and trains the CNN model
├── model_evaluate.py              # Evaluates model performance
├── forgery_cnn_model.h5           # Saved baseline model (no augmentation)
├── forgery_cnn_augmented.keras    # Saved model with image augmentation
├── results/
│   ├── training_plot.png          # Training and validation loss/accuracy
│   ├── confusion_matrix.png       # Confusion matrix
│   ├── classification_report.txt  # Detailed precision/recall/F1 scores
│   └── model_summary.txt          # CNN architecture summary
└── README.md                      # Project documentation

## 📈 Sample Outputs

Here are some visualizations from the model evaluation phase:

- **Confusion Matrix:**

  ![Confusion Matrix](https://github.com/sahooananya/image-forgery-detection/blob/main/results/confusion_matrix.png)

- **Training Accuracy & Loss:**

  ![Training Plot](https://github.com/sahooananya/image-forgery-detection/blob/main/results/training_plot.png)



🔍 Future Work
Implement transfer learning using pretrained models (e.g., MobileNet, EfficientNet).
Improve robustness using adversarial training.
Extend to the localization of tampered regions using segmentation techniques.

📚 References
CASIA v2: http://forensics.idealtest.org/

Columbia Dataset: https://www.ee.columbia.edu/

COVERAGE Dataset: https://github.com/utkarsh-raj/
