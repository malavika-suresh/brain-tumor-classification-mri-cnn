# 🧠 Brain Tumor Detection using CNN

## 🔗 Main Libraries & Tools

- [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange.svg)](https://www.tensorflow.org/) – Deep learning framework for model building  
- [![Keras](https://img.shields.io/badge/Keras-2.12-red.svg)](https://keras.io/) – High-level API for TensorFlow  
- [![OpenCV](https://img.shields.io/badge/OpenCV-4.7.0-blue.svg)](https://opencv.org/) – Image processing and computer vision  
- [![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7.1-brightgreen.svg)](https://matplotlib.org/) – Visualization library for plots & images  
- [![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-yellow.svg)](https://scikit-learn.org/stable/) – Data splitting and evaluation metrics  
- [![imutils](https://img.shields.io/badge/imutils-0.5.4-lightgrey.svg)](https://github.com/jrosebr1/imutils) – Helper functions for image processing  

---

## 📂 Project Structure
```
BRAIN_TUMOR_DETECTION/
│── brain_tumor_dataset/
│   ├── no/                # MRI images without tumor
│   └── yes/               # MRI images with tumor
│── cnn-parameters-improvement-24-0.86.model   # Trained CNN weights
│── data_aug.py            # Script for data augmentation
│── final_rslt.py          # Run inference on a single MRI image
│── ver1_train.py          # Train & evaluate CNN model
```

---

## 🛠️ Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
```

**requirements.txt**
```
tensorflow
numpy
matplotlib
opencv-python
imutils
scikit-learn
```

---

## 🚀 How to Use

1. **Prepare Dataset**  
   Organize your data as:
   ```
   brain_tumor_dataset/
   ├── no/
   └── yes/
   ```
2. **Augment Data (optional)**  
   ```bash
   python data_aug.py
   ```
3. **Train Model**  
   ```bash
   python ver1_train.py
   ```  
   This will generate `cnn-parameters-improvement-24-0.86.model`.
4. **Run Inference on a New Image**  
   Edit `final_rslt.py` to point to your test image, then run:
   ```bash
   python final_rslt.py
   ```

---
## 📊 Model Performance (from paper)

```
Accuracy (Test): 0.95  
F1 Score (Test): 0.93
```

---

## 🧮 How It Works
- **Preprocessing**: grayscale conversion → Gaussian blur → thresholding → morphological cleanup → contour cropping.  
- **CNN Model**: convolutional layers → batch norm & pooling → dense classification head.  
- **Prediction Logic**: outputs probability; if > 0.6 → “Brain Tumor Detected,” else “Normal.”

---

## 📜 Reference & Citation

Please cite our peer-reviewed work if you use this repository:

**Brain Tumour Detection Using Deep Learning**  
[ResearchGate Publication](https://www.researchgate.net/publication/352148333_Brain_Tumour_Detection_Using_Deep_Learning)

---
