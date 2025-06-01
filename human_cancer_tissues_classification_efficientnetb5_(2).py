# streamlit_app.py

import os
import json
import zipfile
import requests
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from tqdm import tqdm

import streamlit as st
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# ─────────────────────────────────────────────────────────────────────────────
# 1. Streamlit Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cancer Tissue Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="auto",
)

st.title("🩺 Human Cancer Tissues Classification (EfficientNetB5)")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Sidebar: Dataset Download & Extraction
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("1. Download & Prepare Dataset")

dataset_slug = st.sidebar.text_input(
    "Kaggle Dataset (owner/dataset)", value="imrankhan77/nct-crc-he-100k"
)

if st.sidebar.button("Download & Unzip Dataset"):
    with st.spinner("Downloading and extracting dataset..."):
        # 2.1 Load Kaggle credentials from ~/.kaggle/kaggle.json
        kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
        if not os.path.exists(kaggle_json_path):
            st.error("🚨 Please upload your `kaggle.json` to ~/.kaggle/kaggle.json first.")
            st.stop()

        with open(kaggle_json_path, "r") as f:
            creds = json.load(f)
        KAGGLE_USERNAME = creds["username"]
        KAGGLE_KEY = creds["key"]

        # 2.2 Build download URL
        owner_slug, slug = dataset_slug.split("/")
        base_url = f"https://www.kaggle.com/api/v1/datasets/download/{owner_slug}/{slug}"
        download_url = base_url + "?accept=true"

        # 2.3 Stream download to a local ZIP file
        session = requests.Session()
        session.auth = (KAGGLE_USERNAME, KAGGLE_KEY)
        response = session.get(download_url, stream=True)
        if response.status_code != 200:
            st.error(f"🚨 Failed to start download (status code {response.status_code}).")
            st.stop()

        total_size = int(response.headers.get("Content-Length", 0))
        zip_path = "nct-crc-he-100k.zip"
        with open(zip_path, "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size=1024), total=total_size // 1024, unit="KB"):
                f.write(chunk)

        # 2.4 Extract into `nct-crc-he-100k/`
        extract_path = "nct-crc-he-100k"
        os.makedirs(extract_path, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in tqdm(zf.namelist(), desc="Extracting files", unit="file"):
                zf.extract(member, extract_path)

        st.success("✅ Dataset downloaded and extracted to `./nct-crc-he-100k/NCT-CRC-HE-100K/`")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Sidebar: Create Train/Val/Test Split
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("2. Create Train/Val/Test Split")

data_dir = "nct-crc-he-100k/NCT-CRC-HE-100K"
if os.path.isdir(data_dir):
    if st.sidebar.button("Generate Splits"):
        with st.spinner("Generating CSV splits..."):
            filepaths = []
            labels = []
            for cls_folder in os.listdir(data_dir):
                cls_path = os.path.join(data_dir, cls_folder)
                if os.path.isdir(cls_path):
                    for fname in os.listdir(cls_path):
                        fpath = os.path.join(cls_path, fname)
                        if os.path.isfile(fpath):
                            filepaths.append(fpath)
                            labels.append(cls_folder)

            df = pd.DataFrame({"filepaths": filepaths, "labels": labels})
            train_df, temp_df = train_test_split(
                df, train_size=0.8, stratify=df["labels"], random_state=123
            )
            val_df, test_df = train_test_split(
                temp_df, train_size=0.5, stratify=temp_df["labels"], random_state=123
            )

            train_df.to_csv("train_df.csv", index=False)
            val_df.to_csv("val_df.csv", index=False)
            test_df.to_csv("test_df.csv", index=False)
        st.success("✅ CSV splits created: `train_df.csv`, `val_df.csv`, `test_df.csv`")
else:
    st.sidebar.warning("Dataset folder not found. Download & unzip first.")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Sidebar: Model Training
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("3. Train Model")

epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=100, value=10)
batch_size = st.sidebar.selectbox("Batch Size", options=[32, 64, 128], index=2)
learning_rate = st.sidebar.number_input("Learning Rate", format="%.5f", value=0.001)

if (
    st.sidebar.button("Start Training")
    and os.path.exists("train_df.csv")
    and os.path.exists("val_df.csv")
    and os.path.exists("test_df.csv")
):
    st.sidebar.write("🔄 Training in progress…")
    train_df = pd.read_csv("train_df.csv")
    val_df = pd.read_csv("val_df.csv")
    test_df = pd.read_csv("test_df.csv")

    img_size = (224, 224)
    class_count = train_df["labels"].nunique()

    tr_gen = ImageDataGenerator()
    ts_gen = ImageDataGenerator()

    train_gen = tr_gen.flow_from_dataframe(
        dataframe=train_df,
        x_col="filepaths",
        y_col="labels",
        target_size=img_size,
        class_mode="categorical",
        shuffle=True,
        batch_size=batch_size,
    )
    val_gen = ts_gen.flow_from_dataframe(
        dataframe=val_df,
        x_col="filepaths",
        y_col="labels",
        target_size=img_size,
        class_mode="categorical",
        shuffle=True,
        batch_size=batch_size,
    )
    test_gen = ts_gen.flow_from_dataframe(
        dataframe=test_df,
        x_col="filepaths",
        y_col="labels",
        target_size=img_size,
        class_mode="categorical",
        shuffle=False,
        batch_size=batch_size,
    )

    # Build model
    base_model = tf.keras.applications.efficientnet.EfficientNetB5(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3), pooling="max"
    )
    base_model.trainable = False

    model = Sequential(
        [
            base_model,
            BatchNormalization(),
            Dense(
                256,
                kernel_regularizer=regularizers.l2(0.016),
                activity_regularizer=regularizers.l1(0.006),
                bias_regularizer=regularizers.l1(0.006),
                activation="relu",
            ),
            Dropout(rate=0.45, seed=123),
            Dense(class_count, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adamax(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Train
    history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, verbose=1)

    # Evaluate on test
    loss_test, acc_test = model.evaluate(test_gen, verbose=0)
    st.sidebar.success(f"Test Loss: {loss_test:.4f} | Test Accuracy: {acc_test:.4f}")

    # Plot metrics
    st.subheader("Training & Validation Metrics")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = list(range(1, epochs + 1))

    ax[0].plot(epochs_range, history.history["loss"], label="Train Loss")
    ax[0].plot(epochs_range, history.history["val_loss"], label="Val Loss")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    ax[1].plot(epochs_range, history.history["accuracy"], label="Train Acc")
    ax[1].plot(epochs_range, history.history["val_accuracy"], label="Val Acc")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()

    st.pyplot(fig)

    # Confusion matrix on test set
    st.subheader("Confusion Matrix & Classification Report")
    preds = model.predict(test_gen)
    y_pred = np.argmax(preds, axis=1)
    cm = confusion_matrix(test_gen.classes, y_pred)
    class_labels = list(train_gen.class_indices.keys())

    fig2, ax2 = plt.subplots(figsize=(8, 8))
    im = ax2.imshow(cm, cmap=plt.cm.Blues)
    ax2.set_xticks(np.arange(len(class_labels)))
    ax2.set_yticks(np.arange(len(class_labels)))
    ax2.set_xticklabels(class_labels, rotation=45, ha="right")
    ax2.set_yticklabels(class_labels)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        ax2.text(j, i, cm[i, j], ha="center", va="center", color=color)
    ax2.set_xlabel("Predicted Label")
    ax2.set_ylabel("True Label")
    fig2.colorbar(im)
    st.pyplot(fig2)

    report = classification_report(test_gen.classes, y_pred, target_names=class_labels, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    # Save model and class indices
    model.save("Model.h5")
    with open("class_indices.json", "w") as f:
        json.dump(train_gen.class_indices, f)
    st.sidebar.success("✅ Model and class_indices.json saved.")
else:
    st.sidebar.info("Upload data, generate splits, then click 'Start Training'")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Main Area: Inference on Uploaded Image
# ─────────────────────────────────────────────────────────────────────────────
st.header("4. Upload an Image for Inference")
st.write("After training, upload a new histopathology image to see its predicted class.")

uploaded_img = st.file_uploader(label="Choose a JPG/PNG", type=["jpg", "jpeg", "png"])
if uploaded_img:
    try:
        img = Image.open(uploaded_img).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)
    except Exception as e:
        st.error(f"❌ Invalid image: {e}")
        st.stop()

    # Load saved model & class indices if not already loaded
    if "inference_model" not in st.session_state:
        if os.path.exists("Model.h5") and os.path.exists("class_indices.json"):
            model_inf = tf.keras.models.load_model("Model.h5")
            ci = json.load(open("class_indices.json"))
            idx_to_class = {v: k for k, v in ci.items()}
            st.session_state["inference_model"] = model_inf
            st.session_state["idx_to_class"] = idx_to_class
        else:
            st.warning("⚠️ `Model.h5` or `class_indices.json` not found. Train first.")
            st.stop()

    model_inf = st.session_state["inference_model"]
    idx_to_class = st.session_state["idx_to_class"]

    # Preprocess & predict
    img_resized = img.resize((224, 224))
    arr = np.array(img_resized).astype("float32")
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[2] == 4:
        arr = arr[..., :3]
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    preds = model_inf.predict(arr, verbose=0)[0]
    top3 = sorted(list(enumerate(preds)), key=lambda x: x[1], reverse=True)[:3]

    st.subheader("Top‐3 Predictions")
    for idx, prob in top3:
        class_name = idx_to_class[idx]
        st.write(f"• **{class_name}**: {prob * 100:.2f}%")

    if st.checkbox("Show Full Probabilities"):
        probs = [(idx_to_class[i], float(preds[i])) for i in range(len(preds))]
        probs_df = pd.DataFrame(probs, columns=["Class", "Probability"])
        probs_df["Probability"] = probs_df["Probability"].apply(lambda x: f"{x * 100:.2f}%")
        st.dataframe(probs_df, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.write("© 2025 Cancer Tissue Classification Demo using EfficientNetB5")
