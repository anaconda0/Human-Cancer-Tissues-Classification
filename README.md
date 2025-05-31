# Human Cancer Tissues Classification with EfficientNetB5

Welcome! This repository contains a Jupyter Notebook that demonstrates how to build, train, and evaluate a histopathology image classifier using **EfficientNetB5** as a base model. The goal is to accurately classify human cancer tissue patches into one of several tissue types.

Below is a guide that explains:

1. **What this project is about**  
2. **Where to get the dataset**  
3. **How to open and run the notebook**  
4. **Dependencies & environment setup**  
5. **Directory structure & file descriptions**

---

## 📖 1. Project Overview

In digital pathology, accurately identifying different tissue types (e.g., tumor epithelium, stroma, lymphocytes, necrosis, etc.) is critical for cancer diagnosis and research. Manually labeling millions of image patches is tedious and time‐consuming. Deep learning, particularly CNNs, can automate this process at scale.

This project’s objectives:

- **Load & preprocess** a large histopathology image dataset.  
- **Build** a CNN classifier by leveraging a pretrained EfficientNetB5 as a feature extractor.  
- **Train & fine‐tune** the top layers (and optionally some EfficientNetB5 layers) on our specific classes.  
- **Evaluate** performance in terms of loss and accuracy.  
- **Provide** an easy‐to‐follow Jupyter Notebook that you can clone, tweak, and rerun on your own.

---

## 📥 2. Dataset

We use the **NCT-CRC-HE-100K** dataset hosted on Kaggle. It contains 100,000 non‐overlapping histological image patches (224×224 pixels) of human colorectal cancer tissue, divided into 9 classes:

- **Adipose (ADI)**  
- **Background (BACK)**  
- **Debris (DEB)**  
- **Lymphocytes (LYM)**  
- **Mucus (MUC)**  
- **Smooth Muscle (MUS)**  
- **Normal Colon Mucosa (NORM)**  
- **Cancer‐associated Stroma (STR)**  
- **Tumor Epithelium (TUM)**  

**Kaggle link to download:**  
https://www.kaggle.com/datasets/imrankhan77/nct-crc-he-100k/data

### How to download

1. Sign in to Kaggle.  
2. Visit the link above.  
3. Click “Download” to get a ZIP (or TAR.GZ) file.  
4. Unzip it to a local directory, for example:
   ```
   datasets/
   └── NCT-CRC-HE-100K/
       ├── ADI/
       ├── BACK/
       ├── DEB/
       ├── LYM/
       ├── MUC/
       ├── MUS/
       ├── NORM/
       ├── STR/
       └── TUM/
   ```
5. Each subfolder contains PNG image patches for that tissue class.

---

## 📝 3. How to Open & Run This Notebook

1. **Clone or download** this repository:
   ```bash
   git clone https://github.com/yourusername/cancer-tissue-classifier.git
   cd cancer-tissue-classifier
   ```
   Or download the ZIP and extract it.

2. **Create a Python 3.7+ virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate       # macOS/Linux
   venv\Scripts\activate        # Windows
   pip install --upgrade pip
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   If no `requirements.txt` exists, install manually:
   ```bash
   pip install jupyterlab
   pip install tensorflow      # or tensorflow-cpu
   pip install matplotlib
   pip install pandas
   pip install pillow
   ```

4. **Launch Jupyter Notebook**:
   ```bash
   jupyter lab
   ```
   or
   ```bash
   jupyter notebook
   ```

5. **Open** the notebook file:
   ```
   human_cancer_tissues_classification_efficientnetb5.ipynb
   ```

6. **Follow each cell** in order:
   - **Configuration** (imports, image size, batch size).  
   - **Load & preprocess data** (using `ImageDataGenerator` or `tf.data.Dataset`).  
   - **Create the EfficientNetB5-based model**.  
   - **Compile** the model (optimizer, learning rate, loss, metrics).  
   - **Train** (`model.fit` with optional callbacks).  
   - **Evaluate** (view training & validation loss/accuracy, test performance).  
   - **Visualize** results (plots of accuracy/loss, sample predictions).

---

## 🛠 4. Dependencies & Environment

Below are recommended Python packages (and approximate versions) needed:

```
python              >= 3.7
tensorflow          >= 2.8.0      # or tensorflow-gpu if you have a CUDA-enabled GPU
numpy               >= 1.19.0
matplotlib          >= 3.3.0
pandas              >= 1.1.0
scikit-learn        >= 0.24.0      # (optional, for extra metrics)
jupyterlab          >= 3.0.0
Pillow              >= 8.0.0
```

You can generate a `requirements.txt` by running:
```bash
pip freeze > requirements.txt
```

---

## 📂 5. Directory Structure & File Descriptions

```
cancer-tissue-classifier/
├── datasets/                             # Local copy of NCT-CRC-HE-100K
│   └── NCT-CRC-HE-100K/
│       ├── ADI/
│       ├── BACK/
│       ├── DEB/
│       ├── LYM/
│       ├── MUC/
│       ├── MUS/
│       ├── NORM/
│       ├── STR/
│       └── TUM/
├── human_cancer_tissues_classification_efficientnetb5.ipynb
├── model.h5                              # (Optional) Saved Keras model
├── class_indices.json                    # (Optional) JSON mapping "class_name"→index
├── app.py                                # (Optional) Streamlit inference app
├── requirements.txt
└── README.md                             # ← This file
```

- **datasets/NCT-CRC-HE-100K/**: Downloaded Kaggle dataset. Each subfolder contains 22,500+ images of size 224×224.  
- **human_cancer_tissues_classification_efficientnetb5.ipynb**: The main Jupyter Notebook that:
  1. Loads the data (train/validation generators).  
  2. Builds an EfficientNetB5-based classifier.  
  3. Trains and evaluates the model.  
  4. Plots relevant metrics.  
- **model.h5** (Optional): A pre-trained Keras model file.  
- **class_indices.json** (Optional): JSON mapping of class names → integer indices, e.g.:  
  ```json
  {
    "ADI": 0,
    "BACK": 1,
    "DEB": 2,
    "LYM": 3,
    "MUC": 4,
    "MUS": 5,
    "NORM": 6,
    "STR": 7,
    "TUM": 8
  }
  ```
- **app.py** (Optional): A Streamlit application that loads the saved model and lets you upload a histology image for inference.  
- **requirements.txt**: Lists all Python dependencies.  
- **README.md**: This readme.

---

## 🎯 6. Tips & Next Steps

1. **Fine‐Tune EfficientNetB5**  
   - After training the top layers, try unfreezing the last blocks of EfficientNetB5 with a **small learning rate** (e.g., `1e-5`). Fine‐tuning often yields higher accuracy on domain-specific images.

2. **Stronger Data Augmentation**  
   - Add more aggressive augmentations (rotations up to 180°, random crops, color jitter, etc.) to improve generalization.  
   - Use `tf.keras.preprocessing.image.ImageDataGenerator` or `tf.data.Dataset` for on‐the‐fly augmentation.

3. **Address Class Imbalance**  
   - Check if some classes have far fewer samples. If imbalance exists, apply `class_weight` in `model.fit(...)` or oversample minority classes.

4. **Plot Confusion Matrix & Classification Report**  
   - After training, use `sklearn.metrics` to compute a confusion matrix and per‐class precision/recall/F1‐score. This helps identify which tissue types the model confuses most.

5. **Save & Load Models**  
   - During training, save the best model via a checkpoint callback:
     ```python
     from tensorflow.keras.callbacks import ModelCheckpoint

     checkpoint = ModelCheckpoint(
         filepath='best_model.h5',
         monitor='val_accuracy',
         save_best_only=True,
         mode='max'
     )

     history = model.fit(
         train_gen,
         epochs=20,
         validation_data=val_gen,
         callbacks=[checkpoint, ...]
     )
     ```
   - Later, load with:
     ```python
     model = tf.keras.models.load_model('best_model.h5')
     ```

6. **Use Mixed Precision (GPU)**  
   - If you have an NVIDIA GPU (e.g., T4, V100), enabling mixed precision speeds up training:
     ```python
     from tensorflow.keras import mixed_precision
     policy = mixed_precision.Policy('mixed_float16')
     mixed_precision.set_global_policy(policy)
     ```
   - Use a suitable optimizer (e.g., `tf.keras.optimizers.Adam(lr=...)`) and consider a loss scale if needed.

---

## 🎉 7. Acknowledgments & References

- **Kaggle Dataset**: “NCT-CRC-HE-100K” by imrankhan77  
  https://www.kaggle.com/datasets/imrankhan77/nct-crc-he-100k/data  
- **EfficientNet Paper**: Mingxing Tan and Quoc V. Le, “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,” ICML 2019.  
- **TensorFlow & Keras Documentation** for guidance on pretrained models, data generators, and training callbacks.

---

Thank you for exploring this project! If you have any questions, run into issues, or want to propose improvements, feel free to open an issue or submit a pull request. Happy coding and happy histopathology classification! 🚀
