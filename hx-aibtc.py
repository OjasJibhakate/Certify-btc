"""
HXAI-BTC: Hybrid Explainable AI for Brain Tumor Classification
=================================================================
Novel Contributions:
  1. EfficientNetB4 + CBAM Attention Module (custom hybrid architecture)
  2. Multi-XAI Fusion: Grad-CAM + LIME + SHAP consensus heatmap
  3. Uncertainty Quantification via Monte Carlo Dropout
  4. 4-class MRI classification (Glioma, Meningioma, Pituitary, No Tumor)

Datasets:
  Primary   — Nickparvar Brain Tumor MRI (4-class, ~7023 images)
              https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
  Secondary — Figshare Brain Tumor MRI (3-class, ~3064 images) for cross-dataset eval
              https://www.kaggle.com/datasets/denizkavi1/brain-tumor

Authors: [Your Name], [Co-authors]
Institution: Symbiosis Institute of Technology, Nagpur
"""

# ============================================================
# SECTION 0: INSTALL & IMPORTS
# ============================================================
# Run this in Colab first:
# !pip install lime shap grad-cam opencv-python-headless scipy

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import cv2
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                         ModelCheckpoint)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)
from sklearn.utils.class_weight import compute_class_weight

import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import shap

print("TF version:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))

# Mixed precision for speed
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# ============================================================
# SECTION 1: CONFIGURATION
# ============================================================

class Config:
    # ── Primary Dataset (Nickparvar) ─────────────────────────────────────
    # Download from Kaggle, upload to Drive, keep this structure:
    #   MyDrive/
    #     nickparvar/
    #       Training/
    #         glioma/       ← folder name must match class name exactly
    #         meningioma/
    #         notumor/
    #         pituitary/
    #       Testing/
    #         glioma/
    #         meningioma/
    #         notumor/
    #         pituitary/
    NICK_TRAIN_DIR = '/content/drive/MyDrive/nickparvar/Training'
    NICK_TEST_DIR  = '/content/drive/MyDrive/nickparvar/Testing'

    # ── Secondary Dataset (Figshare — ashkhagan/figshare-brain-tumor-dataset) ─
    # Download from Kaggle, upload to Drive with this structure:
    #   MyDrive/
    #     figshare/
    #       1.jpg           ← ALL 3064 images in ONE flat folder
    #       2.jpg
    #       ...
    #       3064.jpg
    #       cvind.mat       ← MATLAB label file (MUST be here)
    #       readme          ← ignore this
    #
    # DO NOT create subfolders — the code reads cvind.mat and
    # organizes into class subfolders in /tmp automatically.
    FIGSHARE_DIR   = '/content/drive/MyDrive/figshare'

    # ── Output ───────────────────────────────────────────────────────────
    SAVE_DIR    = '/content/drive/MyDrive/HXAI_BTC/'

    # ── Model ────────────────────────────────────────────────────────────
    IMG_SIZE    = 224
    BATCH_SIZE  = 32
    SEED        = 42

    # Training - Stage 1 (frozen backbone)
    STAGE1_EPOCHS = 15
    STAGE1_LR     = 1e-3

    # Training - Stage 2 (fine-tune top layers)
    STAGE2_EPOCHS     = 25
    STAGE2_LR         = 5e-5
    FINE_TUNE_LAYERS  = 80   # unfreeze top 80 layers of EfficientNetB4

    # Monte Carlo Dropout
    MC_SAMPLES  = 30

    # Primary dataset — 4 classes
    CLASSES    = ['glioma', 'meningioma', 'notumor', 'pituitary']
    N_CLASSES  = 4

    # Figshare only has these 3 (overlapping with primary)
    FIGSHARE_CLASSES       = ['glioma', 'meningioma', 'pituitary']
    # Their index positions inside CLASSES list above ↑
    FIGSHARE_CLASS_INDICES = [0, 1, 3]

    # What % of Figshare to use for fine-tune in Experiment C
    FIGSHARE_FINETUNE_SPLIT = 0.1   # 10% finetune, 90% test

cfg = Config()
os.makedirs(cfg.SAVE_DIR, exist_ok=True)

# ============================================================
# SECTION 2: DATA PIPELINE
# ============================================================

def build_data_generators(cfg):
    """
    CLAHE-enhanced data pipeline for PRIMARY dataset (Nickparvar).
    Returns train, val, test generators for 4-class classification.
    """

    def clahe_preprocess(img):
        """Apply CLAHE then EfficientNet preprocessing."""
        img = img.astype(np.uint8)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        img = tf.keras.applications.efficientnet.preprocess_input(
            img.astype(np.float32))
        return img

    train_datagen = ImageDataGenerator(
        preprocessing_function=clahe_preprocess,
        validation_split=0.2,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.85, 1.15],
        fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator(preprocessing_function=clahe_preprocess)

    train_gen = train_datagen.flow_from_directory(
        cfg.NICK_TRAIN_DIR,
        target_size=(cfg.IMG_SIZE, cfg.IMG_SIZE),
        batch_size=cfg.BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=cfg.SEED
    )
    val_gen = train_datagen.flow_from_directory(
        cfg.NICK_TRAIN_DIR,
        target_size=(cfg.IMG_SIZE, cfg.IMG_SIZE),
        batch_size=cfg.BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=cfg.SEED
    )
    test_gen = test_datagen.flow_from_directory(
        cfg.NICK_TEST_DIR,
        target_size=(cfg.IMG_SIZE, cfg.IMG_SIZE),
        batch_size=cfg.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    print(f"[Nickparvar] Train: {train_gen.n} | Val: {val_gen.n} | Test: {test_gen.n}")
    return train_gen, val_gen, test_gen


def build_figshare_generators(cfg):
    """
    Figshare dataset pipeline for CROSS-DATASET EVALUATION.

    Dataset structure (ashkhagan/figshare-brain-tumor-dataset):
      figshare/
        1.jpg, 2.jpg, ..., 3064.jpg   ← all images in ONE flat folder
        cvind.mat                      ← MATLAB file containing labels
        readme                         ← text file

    cvind.mat contains a struct array. Each entry has a 'label' field:
      1 = meningioma  |  2 = glioma  |  3 = pituitary
    Image index in the .mat corresponds to filename (1.jpg = index 0, etc.)

    We load the .mat → build a DataFrame of (filepath, label) →
    reorganize into class subfolders on the fly (in /tmp) so that
    Keras ImageDataGenerator can use flow_from_directory normally.

    Two generators returned:
      finetune_gen      — 10% of data (Experiment C: few-shot fine-tune)
      figshare_test_gen — 90% of data (Experiment B & C evaluation)
    """
    import scipy.io
    import shutil
    from sklearn.model_selection import train_test_split

    # ── Step 1: Load labels from cvind.mat ───────────────────────────────
    mat_path = os.path.join(cfg.FIGSHARE_DIR, 'cvind.mat')
    mat      = scipy.io.loadmat(mat_path)

    # The .mat file has a variable 'cvsind' or 'label' depending on version.
    # We try both. Structure: shape (3064, 1), values 1/2/3
    if 'cvsind' in mat:
        raw_labels = mat['cvsind'].flatten()          # shape (3064,)
    elif 'label' in mat:
        raw_labels = mat['label'].flatten()
    else:
        # Fallback: grab first non-metadata key
        keys = [k for k in mat.keys() if not k.startswith('_')]
        raw_labels = mat[keys[0]].flatten()

    # Map MATLAB labels → class names
    # 1=meningioma, 2=glioma, 3=pituitary (confirmed from dataset README)
    label_map = {1: 'meningioma', 2: 'glioma', 3: 'pituitary'}

    # ── Step 2: Build DataFrame of valid (filepath, class) pairs ─────────
    records = []
    img_dir = cfg.FIGSHARE_DIR
    for i, lbl in enumerate(raw_labels):
        img_name = f"{i+1}.jpg"                       # images are 1-indexed
        img_path = os.path.join(img_dir, img_name)
        if os.path.exists(img_path) and int(lbl) in label_map:
            records.append((img_path, label_map[int(lbl)]))

    df = pd.DataFrame(records, columns=['filepath', 'class'])
    print(f"\n[Figshare] Total valid images: {len(df)}")
    print(df['class'].value_counts().to_string())

    # ── Step 3: Reorganize into temp folder with class subfolders ─────────
    # Keras flow_from_directory needs:  root/classname/image.jpg
    tmp_root     = '/tmp/figshare_organized'
    tmp_finetune = '/tmp/figshare_finetune'
    tmp_test     = '/tmp/figshare_test'

    for d in [tmp_root, tmp_finetune, tmp_test]:
        if os.path.exists(d):
            shutil.rmtree(d)   # clean slate every run
        for cls in cfg.FIGSHARE_CLASSES:
            os.makedirs(os.path.join(d, cls), exist_ok=True)

    # 90/10 stratified split
    df_test, df_finetune = train_test_split(
        df,
        test_size=cfg.FIGSHARE_FINETUNE_SPLIT,   # 10% for fine-tune
        stratify=df['class'],
        random_state=cfg.SEED
    )

    def copy_to_dir(dataframe, dest_root):
        for _, row in dataframe.iterrows():
            fname = os.path.basename(row['filepath'])
            dst   = os.path.join(dest_root, row['class'], fname)
            shutil.copy2(row['filepath'], dst)

    print("Organizing Figshare images into temp folders...")
    copy_to_dir(df_finetune, tmp_finetune)
    copy_to_dir(df_test,     tmp_test)
    print(f"  Fine-tune set: {len(df_finetune)} images")
    print(f"  Test set:      {len(df_test)} images")

    # ── Step 4: Build generators from the organized temp folders ──────────
    def clahe_preprocess(img):
        img = img.astype(np.uint8)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        img = tf.keras.applications.efficientnet.preprocess_input(
            img.astype(np.float32))
        return img

    finetune_datagen = ImageDataGenerator(
        preprocessing_function=clahe_preprocess,
        rotation_range=15,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator(preprocessing_function=clahe_preprocess)

    finetune_gen = finetune_datagen.flow_from_directory(
        tmp_finetune,
        target_size=(cfg.IMG_SIZE, cfg.IMG_SIZE),
        batch_size=cfg.BATCH_SIZE,
        class_mode='categorical',
        classes=cfg.FIGSHARE_CLASSES,
        shuffle=True,
        seed=cfg.SEED
    )

    figshare_test_gen = test_datagen.flow_from_directory(
        tmp_test,
        target_size=(cfg.IMG_SIZE, cfg.IMG_SIZE),
        batch_size=cfg.BATCH_SIZE,
        class_mode='categorical',
        classes=cfg.FIGSHARE_CLASSES,
        shuffle=False
    )

    print(f"\n[Figshare] Fine-tune gen: {finetune_gen.n} | Test gen: {figshare_test_gen.n}")
    print(f"  Class indices: {figshare_test_gen.class_indices}")
    return finetune_gen, figshare_test_gen


# ============================================================
# SECTION 3: NOVEL ARCHITECTURE — EfficientNetB4 + CBAM
# ============================================================

def channel_attention(x, ratio=16):
    """
    Channel Attention Module (part of CBAM).
    Learns 'what' features to emphasize.
    """
    channels = x.shape[-1]
    # Squeeze via global pooling (both avg and max)
    avg_pool = layers.GlobalAveragePooling2D()(x)
    max_pool = layers.GlobalMaxPooling2D()(x)

    # Shared MLP
    shared_dense_1 = layers.Dense(channels // ratio, activation='relu',
                                   use_bias=False)
    shared_dense_2 = layers.Dense(channels, use_bias=False)

    avg_out = shared_dense_2(shared_dense_1(avg_pool))
    max_out = shared_dense_2(shared_dense_1(max_pool))

    attention = layers.Add()([avg_out, max_out])
    attention = layers.Activation('sigmoid')(attention)
    attention = layers.Reshape((1, 1, channels))(attention)
    return layers.Multiply()([x, attention])


def spatial_attention(x, kernel_size=7):
    """
    Spatial Attention Module (part of CBAM).
    Learns 'where' to focus in the feature map.
    Critical for localizing tumors in MRI.
    """
    # Channel-wise avg and max
    avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])

    attention = layers.Conv2D(1, kernel_size, padding='same',
                               activation='sigmoid', use_bias=False)(concat)
    return layers.Multiply()([x, attention])


def cbam_block(x, ratio=16, kernel_size=7):
    """
    Full CBAM: Channel Attention → Spatial Attention.
    This is the core novel component of our architecture.
    Paper reference: Woo et al., ECCV 2018 - "CBAM: Convolutional Block Attention Module"
    """
    x = channel_attention(x, ratio)
    x = spatial_attention(x, kernel_size)
    return x


def build_hxai_model(cfg):
    """
    Build HXAI-BTC model:
    EfficientNetB4 backbone → CBAM Attention → Multi-scale fusion → Classification
    """
    inputs = layers.Input(shape=(cfg.IMG_SIZE, cfg.IMG_SIZE, 3))

    # Backbone: EfficientNetB4 (pretrained ImageNet)
    backbone = EfficientNetB4(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs
    )
    backbone.trainable = False  # Frozen in Stage 1

    # Extract multi-scale features (novel: not just the last layer!)
    # Early features = texture/edges, Late features = semantic/tumor regions
    feat_early = backbone.get_layer('block4a_expand_activation').output   # 28x28
    feat_mid   = backbone.get_layer('block6a_expand_activation').output   # 14x14
    feat_late  = backbone.output                                            # 7x7

    # Apply CBAM to each scale
    feat_early = cbam_block(feat_early)
    feat_mid   = cbam_block(feat_mid)
    feat_late  = cbam_block(feat_late)

    # Global Average Pooling for each scale
    gap_early = layers.GlobalAveragePooling2D()(feat_early)  # (batch, 96)
    gap_mid   = layers.GlobalAveragePooling2D()(feat_mid)    # (batch, 192)
    gap_late  = layers.GlobalAveragePooling2D()(feat_late)   # (batch, 1792)

    # Concatenate multi-scale features
    multi_scale = layers.Concatenate()([gap_early, gap_mid, gap_late])
    # Shape: (batch, 2080)

    # Classification head
    x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(multi_scale)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4)(x)  # MC Dropout — keep this active during inference!

    x = layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)  # MC Dropout

    # Output — float32 for stability with mixed precision
    outputs = layers.Dense(cfg.N_CLASSES, activation='softmax',
                            dtype='float32')(x)

    model = Model(inputs=inputs, outputs=outputs, name='HXAI_BTC')
    return model, backbone


# ============================================================
# SECTION 4: TRAINING
# ============================================================

def get_class_weights(train_gen):
    """Compute class weights to handle imbalance."""
    labels = train_gen.classes
    weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return dict(enumerate(weights))


def compile_model(model, lr):
    model.compile(
        optimizer=AdamW(learning_rate=lr, weight_decay=1e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy',
                 tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_acc')]
    )


def get_callbacks(cfg, stage):
    return [
        ModelCheckpoint(
            f"{cfg.SAVE_DIR}best_hxai_btc_stage{stage}.keras",
            monitor='val_accuracy', save_best_only=True, verbose=1
        ),
        EarlyStopping(monitor='val_accuracy', patience=7,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, min_lr=1e-7, verbose=1)
    ]


def train_model(cfg, train_gen, val_gen):
    model, backbone = build_hxai_model(cfg)
    model.summary()

    class_weights = get_class_weights(train_gen)
    print("Class weights:", class_weights)

    # ── STAGE 1: Train head only (backbone frozen) ──────────────────────
    print("\n" + "="*60)
    print("STAGE 1: Training classification head (backbone frozen)")
    print("="*60)
    compile_model(model, cfg.STAGE1_LR)
    hist1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=cfg.STAGE1_EPOCHS,
        class_weight=class_weights,
        callbacks=get_callbacks(cfg, 1)
    )

    # ── STAGE 2: Fine-tune top layers ────────────────────────────────────
    print("\n" + "="*60)
    print(f"STAGE 2: Fine-tuning top {cfg.FINE_TUNE_LAYERS} layers")
    print("="*60)

    # Unfreeze top N layers of backbone
    for layer in backbone.layers[-cfg.FINE_TUNE_LAYERS:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    compile_model(model, cfg.STAGE2_LR)
    hist2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=cfg.STAGE2_EPOCHS,
        class_weight=class_weights,
        callbacks=get_callbacks(cfg, 2)
    )

    return model, hist1, hist2


# ============================================================
# SECTION 5: EVALUATION
# ============================================================

def full_evaluation(model, test_gen, cfg):
    """Standard evaluation metrics."""
    print("\n" + "="*60)
    print("EVALUATION ON TEST SET")
    print("="*60)

    results = model.evaluate(test_gen, verbose=1)
    print(f"\nTest Loss:     {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]*100:.2f}%")
    print(f"Top-2 Accuracy: {results[2]*100:.2f}%")

    # Predictions
    test_gen.reset()
    y_pred_prob = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = test_gen.classes

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred,
                                 target_names=cfg.CLASSES, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=cfg.CLASSES, yticklabels=cfg.CLASSES)
    plt.title('HXAI-BTC Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"{cfg.SAVE_DIR}confusion_matrix.png", dpi=150)
    plt.show()

    # ROC-AUC
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
    auc = roc_auc_score(y_true_bin, y_pred_prob, multi_class='ovr', average='macro')
    print(f"\nMacro AUC-ROC: {auc:.4f}")

    return y_pred_prob, y_pred, y_true


# ============================================================
# SECTION 6: UNCERTAINTY QUANTIFICATION (Monte Carlo Dropout)
# ============================================================

def mc_dropout_predict(model, img_array, n_samples=30):
    """
    Monte Carlo Dropout Inference.
    Run forward pass N times WITH dropout active to estimate uncertainty.
    Lower variance = model is confident.
    Higher variance = model is uncertain (flag for human review!).
    """
    # img_array shape: (1, H, W, 3)
    preds = np.stack([
        model(img_array, training=True)  # training=True keeps dropout ON
        for _ in range(n_samples)
    ], axis=0)  # (n_samples, 1, n_classes)

    mean_pred   = preds.mean(axis=0)[0]   # (n_classes,)
    std_pred    = preds.std(axis=0)[0]    # (n_classes,) — uncertainty
    entropy     = -np.sum(mean_pred * np.log(mean_pred + 1e-8))  # predictive entropy

    return mean_pred, std_pred, entropy


def visualize_uncertainty(model, img_tensor, img_display, cfg, true_label=None):
    """Visualize prediction with uncertainty bars."""
    mean_pred, std_pred, entropy = mc_dropout_predict(
        model, img_tensor, cfg.MC_SAMPLES)

    pred_class = np.argmax(mean_pred)
    confidence = mean_pred[pred_class]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Image
    axes[0].imshow(img_display)
    title = f"Predicted: {cfg.CLASSES[pred_class]} ({confidence*100:.1f}%)"
    if true_label is not None:
        title += f"\nTrue: {cfg.CLASSES[true_label]}"
    axes[0].set_title(title, fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Uncertainty bars
    colors = ['#e74c3c' if i == pred_class else '#3498db' for i in range(cfg.N_CLASSES)]
    bars = axes[1].bar(cfg.CLASSES, mean_pred, yerr=std_pred,
                        color=colors, alpha=0.8, capsize=5, edgecolor='black')
    axes[1].set_ylabel('Probability')
    axes[1].set_title(f'Prediction Uncertainty\n(Entropy: {entropy:.4f})', fontsize=12)
    axes[1].set_ylim(0, 1.1)
    axes[1].tick_params(axis='x', rotation=15)

    # Risk flag
    risk_text = "⚠ HIGH UNCERTAINTY — Flag for Expert Review" if entropy > 0.5 \
                else "✅ HIGH CONFIDENCE Prediction"
    fig.suptitle(risk_text, fontsize=11, color='red' if entropy > 0.5 else 'green')

    plt.tight_layout()
    plt.savefig(f"{cfg.SAVE_DIR}uncertainty_visualization.png", dpi=150)
    plt.show()

    return mean_pred, std_pred, entropy


# ============================================================
# SECTION 7: GRAD-CAM (Enhanced)
# ============================================================

def get_gradcam_heatmap(model, img_array, layer_name, pred_index=None):
    """Generate Grad-CAM heatmap."""
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(original_img, heatmap, alpha=0.4):
    """Overlay Grad-CAM heatmap on original image."""
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    superimposed = cv2.addWeighted(
        original_img.astype(np.uint8), 1 - alpha,
        heatmap_colored, alpha, 0)
    return superimposed


# ============================================================
# SECTION 8: LIME EXPLANATION
# ============================================================

def get_lime_explanation(model, img_array_raw, n_samples=1000):
    """
    LIME: Local Interpretable Model-Agnostic Explanations.
    Shows which superpixel regions the model focused on.
    """
    explainer = lime_image.LimeImageExplainer()

    def predict_fn(images):
        processed = np.array([
            tf.keras.applications.efficientnet.preprocess_input(
                img.astype(np.float32))
            for img in images
        ])
        return model.predict(processed, verbose=0)

    explanation = explainer.explain_instance(
        img_array_raw.astype(np.double),
        predict_fn,
        top_labels=4,
        hide_color=0,
        num_samples=n_samples,
        batch_size=32
    )
    return explanation


def visualize_lime(explanation, img_raw, pred_class, cfg):
    """Visualize LIME explanation for predicted class."""
    temp, mask = explanation.get_image_and_mask(
        pred_class,
        positive_only=False,
        num_features=10,
        hide_rest=False
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_raw); axes[0].set_title('Original MRI'); axes[0].axis('off')

    # Positive regions only
    temp_pos, mask_pos = explanation.get_image_and_mask(
        pred_class, positive_only=True, num_features=5, hide_rest=True)
    axes[1].imshow(mark_boundaries(temp_pos / 255.0, mask_pos))
    axes[1].set_title('LIME: Supporting Regions (Green)', fontweight='bold')
    axes[1].axis('off')

    # All regions
    axes[2].imshow(mark_boundaries(temp / 255.0, mask))
    axes[2].set_title(f'LIME: Full Explanation\n({cfg.CLASSES[pred_class]})', fontweight='bold')
    axes[2].axis('off')

    plt.suptitle('LIME Explanation — HXAI-BTC', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{cfg.SAVE_DIR}lime_explanation.png", dpi=150)
    plt.show()


# ============================================================
# SECTION 9: SHAP EXPLANATION
# ============================================================

def get_shap_explanation(model, img_tensor, background_samples):
    """
    SHAP Deep Explainer.
    Shows pixel-level contribution to each class prediction.
    """
    explainer = shap.DeepExplainer(model, background_samples)
    shap_values = explainer.shap_values(img_tensor)
    return shap_values


def visualize_shap(shap_values, img_array, cfg, pred_class):
    """Visualize SHAP values."""
    fig, axes = plt.subplots(1, cfg.N_CLASSES + 1, figsize=(20, 4))

    axes[0].imshow(img_array[0].astype(np.uint8))
    axes[0].set_title('Original MRI', fontweight='bold'); axes[0].axis('off')

    for i, cls_name in enumerate(cfg.CLASSES):
        shap_img = shap_values[i][0]  # (H, W, C)
        shap_mean = np.mean(np.abs(shap_img), axis=-1)  # collapse channels
        border = 3 if i == pred_class else 0
        axes[i+1].imshow(img_array[0].astype(np.uint8), alpha=0.5)
        axes[i+1].imshow(shap_mean, cmap='hot', alpha=0.6)
        title = f'{cls_name}\n{"← PREDICTED" if i == pred_class else ""}'
        axes[i+1].set_title(title, fontweight='bold' if i == pred_class else 'normal',
                             color='red' if i == pred_class else 'black')
        axes[i+1].axis('off')

    plt.suptitle('SHAP Deep Explainer — Pixel Importance per Class',
                  fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{cfg.SAVE_DIR}shap_explanation.png", dpi=150)
    plt.show()


# ============================================================
# SECTION 10: CONSENSUS HEATMAP (Novel Contribution)
# ============================================================

def generate_consensus_heatmap(gradcam_heatmap, lime_mask, shap_heatmap,
                                 original_img, pred_class, cfg):
    """
    NOVEL CONTRIBUTION: Consensus Heatmap
    ======================================
    Fuses Grad-CAM + LIME + SHAP into a single consensus map.
    Regions highlighted by ALL THREE methods = most reliable explanations.
    This addresses the common criticism that individual XAI methods
    can be inconsistent.

    Method:
        consensus = normalize(gradcam) * w1 +
                    normalize(lime)    * w2 +
                    normalize(shap)    * w3
    """
    H, W = original_img.shape[:2]

    # Normalize each to [0, 1]
    def norm(x):
        x = cv2.resize(x.astype(np.float32), (W, H))
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-8)

    gcam = norm(gradcam_heatmap)
    lime_n = norm(lime_mask.astype(np.float32))
    shap_n = norm(np.mean(np.abs(shap_heatmap), axis=-1))

    # Weighted fusion (Grad-CAM gets highest weight as it's architecture-aware)
    w = [0.45, 0.30, 0.25]
    consensus = w[0] * gcam + w[1] * lime_n + w[2] * shap_n
    consensus = (consensus - consensus.min()) / (consensus.max() - consensus.min() + 1e-8)

    # Overlay on original
    consensus_colored = cv2.applyColorMap(
        np.uint8(255 * consensus), cv2.COLORMAP_TURBO)
    consensus_colored = cv2.cvtColor(consensus_colored, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(
        original_img.astype(np.uint8), 0.5,
        consensus_colored, 0.5, 0)

    # Plot comparison
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    imgs = [original_img, gcam, lime_n, shap_n, superimposed]
    titles = ['Original MRI', 'Grad-CAM', 'LIME Mask',
              'SHAP Importance', f'Consensus Map\n({cfg.CLASSES[pred_class]})']
    cmaps = [None, 'jet', 'Greens', 'hot', None]

    for ax, img, title, cmap in zip(axes, imgs, titles, cmaps):
        if cmap:
            ax.imshow(img, cmap=cmap)
        else:
            ax.imshow(img.astype(np.uint8) if img.dtype != np.float32 else img)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')

    plt.suptitle('Multi-XAI Consensus Heatmap — HXAI-BTC',
                  fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{cfg.SAVE_DIR}consensus_heatmap.png", dpi=150,
                bbox_inches='tight')
    plt.show()

    return consensus


# ============================================================
# SECTION 11: TRAINING HISTORY VISUALIZATION
# ============================================================

def plot_training_history(hist1, hist2, cfg):
    """Plot combined training history from both stages."""
    acc1  = hist1.history['accuracy']
    vacc1 = hist1.history['val_accuracy']
    acc2  = hist2.history['accuracy']
    vacc2 = hist2.history['val_accuracy']
    loss1  = hist1.history['loss']
    vloss1 = hist1.history['val_loss']
    loss2  = hist2.history['loss']
    vloss2 = hist2.history['val_loss']

    ep1 = len(acc1)
    ep2 = ep1 + len(acc2)
    x1 = range(1, ep1 + 1)
    x2 = range(ep1 + 1, ep2 + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(x1, acc1, 'b-', label='Train Acc (S1)')
    ax1.plot(x1, vacc1, 'b--', label='Val Acc (S1)')
    ax1.plot(x2, acc2, 'r-', label='Train Acc (S2 FT)')
    ax1.plot(x2, vacc2, 'r--', label='Val Acc (S2 FT)')
    ax1.axvline(x=ep1, color='gray', linestyle=':', label='Fine-tune start')
    ax1.set_title('Accuracy — HXAI-BTC', fontweight='bold')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(x1, loss1, 'b-', label='Train Loss (S1)')
    ax2.plot(x1, vloss1, 'b--', label='Val Loss (S1)')
    ax2.plot(x2, loss2, 'r-', label='Train Loss (S2 FT)')
    ax2.plot(x2, vloss2, 'r--', label='Val Loss (S2 FT)')
    ax2.axvline(x=ep1, color='gray', linestyle=':', label='Fine-tune start')
    ax2.set_title('Loss — HXAI-BTC', fontweight='bold')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle('Training History: EfficientNetB4 + CBAM (HXAI-BTC)',
                  fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{cfg.SAVE_DIR}training_history.png", dpi=150)
    plt.show()


# ============================================================
# SECTION 12: CROSS-DATASET EVALUATION (Novel Contribution)
# ============================================================

def cross_dataset_eval_expB(model, figshare_test_gen, cfg):
    """
    EXPERIMENT B — Zero-shot generalization.
    Model trained ONLY on Nickparvar, tested on Figshare.
    No retraining. Pure generalization test.

    Because Figshare has 3 classes (no notumor), we:
      - Only look at those 3 output neurons
      - Re-normalize their probabilities to sum to 1
      - Evaluate on those 3 classes only
    """
    print("\n" + "="*60)
    print("EXPERIMENT B: Zero-shot Cross-Dataset Evaluation")
    print("(Trained on Nickparvar → Tested on Figshare, NO retraining)")
    print("="*60)

    figshare_test_gen.reset()
    raw_preds = model.predict(figshare_test_gen, verbose=1)
    y_true    = figshare_test_gen.classes  # 0=glioma, 1=meningioma, 2=pituitary

    # Extract only the 3 relevant class columns and renormalize
    relevant = raw_preds[:, cfg.FIGSHARE_CLASS_INDICES]  # cols 0,1,3
    relevant = relevant / relevant.sum(axis=1, keepdims=True)
    y_pred   = np.argmax(relevant, axis=1)

    acc = np.mean(y_pred == y_true)
    print(f"\nZero-shot Test Accuracy (3-class): {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred,
                                 target_names=cfg.FIGSHARE_CLASSES, digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=cfg.FIGSHARE_CLASSES,
                yticklabels=cfg.FIGSHARE_CLASSES)
    plt.title('Exp B — Zero-shot Figshare\n(No Retraining)',
               fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{cfg.SAVE_DIR}expB_zeroshot_confusion.png", dpi=150)
    plt.show()

    return acc, y_pred, y_true


def cross_dataset_eval_expC(model, finetune_gen, figshare_test_gen, cfg):
    """
    EXPERIMENT C — Few-shot fine-tune on Figshare.
    Take the trained model, fine-tune ONLY the classification head
    on 10% of Figshare data, then test on the remaining 90%.

    This shows the model adapts to new scanner/hospital data fast.
    """
    print("\n" + "="*60)
    print("EXPERIMENT C: Few-shot Fine-tune on Figshare (10% data)")
    print("="*60)

    # Freeze everything except the last 2 Dense layers
    for layer in model.layers:
        layer.trainable = False
    for layer in model.layers[-4:]:   # last Dense + BN + Dropout + output
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy']
    )

    # Quick fine-tune — just 5 epochs on the tiny Figshare fine-tune set
    # We need one-hot labels for the 3-class subset
    # Build a tiny wrapper generator that maps 3-class → 4-class one-hot
    class FigshareWrapper(tf.keras.utils.Sequence):
        def __init__(self, gen, class_indices, n_classes):
            self.gen           = gen
            self.class_indices = class_indices  # [0, 1, 3]
            self.n_classes     = n_classes
            self.n             = gen.n

        def __len__(self):
            return len(self.gen)

        def __getitem__(self, idx):
            X, y_3class = self.gen[idx]
            # y_3class is one-hot with 3 cols → expand to 4 cols
            batch = y_3class.shape[0]
            y_4class = np.zeros((batch, self.n_classes), dtype=np.float32)
            for i, col in enumerate(self.class_indices):
                y_4class[:, col] = y_3class[:, i]
            return X, y_4class

    finetune_wrapped = FigshareWrapper(
        finetune_gen, cfg.FIGSHARE_CLASS_INDICES, cfg.N_CLASSES)

    print(f"Fine-tuning on {finetune_gen.n} Figshare samples for 5 epochs...")
    model.fit(
        finetune_wrapped,
        epochs=5,
        verbose=1
    )

    # Now evaluate on the 90% test split
    print("\nEvaluating on Figshare test set (90%)...")
    figshare_test_gen.reset()
    raw_preds = model.predict(figshare_test_gen, verbose=1)
    y_true    = figshare_test_gen.classes

    relevant = raw_preds[:, cfg.FIGSHARE_CLASS_INDICES]
    relevant = relevant / relevant.sum(axis=1, keepdims=True)
    y_pred   = np.argmax(relevant, axis=1)

    acc = np.mean(y_pred == y_true)
    print(f"\nFew-shot Fine-tune Accuracy (3-class): {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred,
                                 target_names=cfg.FIGSHARE_CLASSES, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=cfg.FIGSHARE_CLASSES,
                yticklabels=cfg.FIGSHARE_CLASSES)
    plt.title('Exp C — Few-shot Fine-tune Figshare\n(10% fine-tune → 90% test)',
               fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{cfg.SAVE_DIR}expC_finetune_confusion.png", dpi=150)
    plt.show()

    return acc, y_pred, y_true


def print_cross_dataset_summary(acc_nickparvar, acc_expB, acc_expC):
    """Print the 3-experiment summary table — goes straight into your paper."""
    print("\n" + "="*70)
    print("CROSS-DATASET GENERALIZATION RESULTS (Table for Paper)")
    print("="*70)
    rows = [
        ['Exp A', 'Nickparvar → Nickparvar', '4-class', f'{acc_nickparvar*100:.2f}%',
         'Standard benchmark'],
        ['Exp B', 'Nickparvar → Figshare',   '3-class', f'{acc_expB*100:.2f}%',
         'Zero-shot generalization'],
        ['Exp C', 'Nickparvar+Figshare(10%)', '3-class', f'{acc_expC*100:.2f}%',
         'Few-shot adaptation'],
    ]
    df = pd.DataFrame(rows, columns=['Experiment', 'Train→Test',
                                      'Task', 'Accuracy', 'Description'])
    print(df.to_string(index=False))
    print("\n✅ This 3-experiment table is your main cross-dataset contribution.")
    return df




def print_comparison_table():
    """
    Compare HXAI-BTC against baseline models (fill in your actual results).
    This table goes directly into your research paper.
    """
    data = {
        'Model':          ['MobileNet', 'VGG-19', 'ResNet50', 'InceptionV3',
                           'EfficientNetB4', 'HXAI-BTC (Ours)'],
        'Accuracy (%)':   ['~96', '~94', '~92.5', '~95', '~96.5', '97.XX'],
        'XAI Method':     ['None', 'Grad-CAM', 'Grad-CAM', 'None',
                           'Grad-CAM', 'Grad-CAM+LIME+SHAP\n(Consensus)'],
        'Task':           ['Binary']*5 + ['4-Class'],
        'Uncertainty':    ['No']*5 + ['Yes (MC Dropout)'],
        'Novel Arch':     ['No']*5 + ['Yes (CBAM Attention)'],
    }
    df = pd.DataFrame(data)
    print("\n" + "="*80)
    print("MODEL COMPARISON TABLE (for paper)")
    print("="*80)
    print(df.to_string(index=False))
    return df


# ============================================================
# SECTION 13: FULL PIPELINE (MAIN)
# ============================================================

def main():
    print("🧠 HXAI-BTC: Hybrid Explainable AI for Brain Tumor Classification")
    print("="*60)

    # ── 1. Build data generators ──────────────────────────────────────────
    train_gen, val_gen, nick_test_gen = build_data_generators(cfg)
    finetune_gen, figshare_test_gen   = build_figshare_generators(cfg)

    # ── 2. Train model on Nickparvar ──────────────────────────────────────
    model, hist1, hist2 = train_model(cfg, train_gen, val_gen)

    # ── 3. Training curves ────────────────────────────────────────────────
    plot_training_history(hist1, hist2, cfg)

    # ── 4. EXPERIMENT A: Standard eval on Nickparvar test set ─────────────
    print("\n" + "="*60)
    print("EXPERIMENT A: Standard Nickparvar Test Set Evaluation")
    print("="*60)
    y_pred_prob, y_pred_A, y_true_A = full_evaluation(model, nick_test_gen, cfg)
    acc_A = np.mean(y_pred_A == y_true_A)

    # ── 5. EXPERIMENT B: Zero-shot on Figshare ────────────────────────────
    acc_B, y_pred_B, y_true_B = cross_dataset_eval_expB(
        model, figshare_test_gen, cfg)

    # ── 6. EXPERIMENT C: Few-shot fine-tune on Figshare ───────────────────
    acc_C, y_pred_C, y_true_C = cross_dataset_eval_expC(
        model, finetune_gen, figshare_test_gen, cfg)

    # ── 7. Cross-dataset summary table ───────────────────────────────────
    print_cross_dataset_summary(acc_A, acc_B, acc_C)

    # ── 8. XAI on one sample from Nickparvar test ─────────────────────────
    nick_test_gen.reset()
    sample_batch, label_batch = next(nick_test_gen)
    sample_idx  = 0
    img_tensor  = sample_batch[[sample_idx]]
    true_label  = np.argmax(label_batch[sample_idx])

    img_display = (img_tensor[0] * 0.5 + 0.5) * 255
    img_display = np.clip(img_display, 0, 255).astype(np.uint8)

    pred_probs = model.predict(img_tensor)
    pred_class = np.argmax(pred_probs[0])
    print(f"\nXAI Sample — True: {cfg.CLASSES[true_label]} | "
          f"Predicted: {cfg.CLASSES[pred_class]} "
          f"({pred_probs[0][pred_class]*100:.1f}%)")

    # ── 9. Grad-CAM ───────────────────────────────────────────────────────
    last_conv_layer = 'top_activation'
    heatmap = get_gradcam_heatmap(model, img_tensor, last_conv_layer, pred_class)
    gradcam_overlay = overlay_gradcam(img_display, heatmap)
    plt.figure(figsize=(6, 5))
    plt.imshow(gradcam_overlay)
    plt.title(f'Grad-CAM: {cfg.CLASSES[pred_class]}', fontweight='bold')
    plt.axis('off')
    plt.savefig(f"{cfg.SAVE_DIR}gradcam.png", dpi=150)
    plt.show()

    # ── 10. LIME ──────────────────────────────────────────────────────────
    lime_exp = get_lime_explanation(model, img_display)
    _, lime_mask = lime_exp.get_image_and_mask(
        pred_class, positive_only=True, num_features=5, hide_rest=True)
    visualize_lime(lime_exp, img_display, pred_class, cfg)

    # ── 11. SHAP ──────────────────────────────────────────────────────────
    background   = sample_batch[:5]
    shap_values  = get_shap_explanation(model, img_tensor, background)
    visualize_shap(shap_values, img_tensor, cfg, pred_class)

    # ── 12. Consensus Heatmap (NOVEL CONTRIBUTION) ────────────────────────
    consensus = generate_consensus_heatmap(
        heatmap, lime_mask, shap_values[pred_class][0],
        img_display, pred_class, cfg)

    # ── 13. Uncertainty Quantification ───────────────────────────────────
    visualize_uncertainty(model, img_tensor, img_display, cfg,
                          true_label=true_label)

    # ── 14. Model comparison table ────────────────────────────────────────
    print_comparison_table()

    print("\n✅ HXAI-BTC Full Pipeline Complete!")
    print(f"All outputs saved to: {cfg.SAVE_DIR}")
    return model


# ============================================================
# RUN
# ============================================================
if __name__ == '__main__':
    model = main()