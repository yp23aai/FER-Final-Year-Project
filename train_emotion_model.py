"""
EMOTION DETECTION MODEL TRAINING SCRIPT
Author: Yodhitomo Sidhi Pranoto
Dataset: RAF-DB (Real-world Affective Faces Database)

This script trains a facial expression model using Transfer Learning:
- Base: MobileNetV2 pre-trained on ImageNet (already knows edges/textures/shapes)
- Top: Custom layers trained specifically on RAF-DB emotions

Expected accuracy: 70-80% on RAF-DB (vs ~60-65% from scratch CNN)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2   # Pre-trained model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf

# ========================================
# CONFIGURATION SETTINGS
# ========================================

# Image size - MobileNetV2 works best at 224x224
# (larger than original 100x100 but gives much better accuracy)
IMG_SIZE = 224

# Training parameters
BATCH_SIZE = 32          # Number of images processed at once
EPOCHS = 50              # Maximum number of training iterations
LEARNING_RATE = 0.001    # How fast the model learns

# Dataset paths - run organize_rafdb.py first to set these up
TRAIN_DIR = 'RAF-DB/train'  # Folder with training images (organised by emotion)
TEST_DIR = 'RAF-DB/test'    # Folder with test images (organised by emotion)

# Two-phase training epochs
EPOCHS_PHASE1 = 15  # Phase 1: train only new layers (fast)
EPOCHS_PHASE2 = 25  # Phase 2: fine-tune top of MobileNetV2 (slower but better)

print("="*70)
print("RAF-DB EMOTION DETECTION MODEL TRAINING")
print("Author: Yodhitomo Sidhi Pranoto")
print(f"TensorFlow: {tf.__version__}")
print(f"GPU: {'Available' if len(tf.config.list_physical_devices('GPU')) > 0 else 'Not found (CPU mode)'}")
print("="*70)

# ========================================
# DATA LOADING WITH AUGMENTATION
# ========================================

# Data augmentation for training - creates variations to improve generalization
# This helps the model work better on new, unseen images
train_datagen = ImageDataGenerator(
    rescale=1./255,              # Normalize pixel values to 0-1 range
    rotation_range=15,           # Randomly rotate images up to 15 degrees
    width_shift_range=0.1,       # Randomly shift images horizontally by 10%
    height_shift_range=0.1,      # Randomly shift images vertically by 10%
    shear_range=0.1,             # Random shearing transformation
    zoom_range=0.1,              # Random zoom in/out
    horizontal_flip=True,        # Randomly flip images horizontally (faces are symmetric)
    fill_mode='nearest',         # Fill any empty pixels after transformations
    validation_split=0.15        # Use 15% of training images for validation
)

# For testing, only normalize (no augmentation on test data)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training images (85% of train folder)
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),  # Resize all images to 224x224
    batch_size=BATCH_SIZE,
    color_mode='rgb',                  # Use colour images (MobileNetV2 needs RGB)
    class_mode='categorical',          # One-hot encoded labels for 7 classes
    subset='training',                 # Use 85% for training
    shuffle=True
)

# Load validation images (15% of train folder, used to check for overfitting)
val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='categorical',
    subset='validation',               # Use 15% for validation
    shuffle=False
)

# Load test images (completely unseen during training)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='categorical',
    shuffle=False                      # Don't shuffle so evaluation is consistent
)

# Display dataset information
print(f"\n✓ Training samples:   {train_generator.samples}")
print(f"✓ Validation samples: {val_generator.samples}")
print(f"✓ Test samples:       {test_generator.samples}")
print(f"✓ Emotion classes:    {list(train_generator.class_indices.keys())}\n")

# ========================================
# BUILD MODEL USING TRANSFER LEARNING
# ========================================

def build_emotion_model():
    """
    Builds an emotion model using Transfer Learning with MobileNetV2.
    
    Architecture:
      MobileNetV2 base (frozen) → GlobalAveragePooling → Dense 512 → Dropout → Dense 7
    
    Returns:
        model: Compiled Keras model ready for training
        base_model: Reference to MobileNetV2 (for phase 2 unfreezing)
    """
    
    # ===== LOAD PRE-TRAINED MOBILENETV2 =====
    # weights='imagenet' means use weights pre-trained on ImageNet dataset
    # include_top=False means remove the final 1000-class ImageNet classifier
    # so we can attach our own 7-emotion classifier instead
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)  # 224x224 RGB images
    )
    
    # ===== PHASE 1: FREEZE THE BASE MODEL =====
    # We don't want to change MobileNetV2's weights yet
    # Just train the new emotion-specific layers we're adding on top
    base_model.trainable = False
    print(f"  MobileNetV2 loaded ({len(base_model.layers)} layers, frozen for Phase 1)")
    
    # ===== BUILD THE FULL MODEL =====
    # Connect: Input → MobileNetV2 → Our custom layers → 7 emotions
    
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Pass images through MobileNetV2 feature extractor
    # training=False keeps BatchNorm layers frozen (important!)
    x = base_model(inputs, training=False)
    
    # GlobalAveragePooling: compress feature maps from (7,7,1280) to (1280,)
    # Takes the average of each feature map - efficient and reduces overfitting
    x = GlobalAveragePooling2D()(x)
    
    # Dense layer to learn emotion-specific feature combinations
    x = Dense(512, activation='relu')(x)   # 512 neurons, ReLU activation
    
    # Dropout: randomly disable 30% of neurons during training to prevent overfitting in order to not rely on any single neuron too much
    x = Dropout(0.3)(x)
    
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Output layer: 7 neurons (one per emotion)
    # Softmax converts raw scores to probabilities that sum to 1.0
    # e.g. [0.03, 0.02, 0.01, 0.85, 0.04, 0.03, 0.02] = 85% Happiness
    outputs = Dense(7, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    return model, base_model

# Build the model
print("Building model with MobileNetV2 transfer learning...")
model, base_model = build_emotion_model()

# Compile with Adam optimizer
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',   # Standard loss for multi-class problems
    metrics=['accuracy']
)

print(f"  Total parameters: {model.count_params():,}")
model.summary()

# ========================================
# TRAINING CALLBACKS
# ========================================

callbacks = [
    # Stop training if validation accuracy doesn't improve for 8 epochs
    # restore_best_weights to always end with the best version of the model
    EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    
    # Reduce learning rate when stuck
    # If val_loss doesn't improve for 4 epochs, halve the learning rate
    # (Take smaller steps when getting close to the optimum)
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-7,
        verbose=1
    ),
    
    # Always save the best model seen during training
    ModelCheckpoint(
        'best_emotion_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# ========================================
# PHASE 1 TRAINING (frozen base)
# ========================================

print("\n" + "="*70)
print("PHASE 1: Training new emotion layers (base model frozen)")
print("="*70)
print("Training only the new top layers. Fast - ~10-15 minutes.\n")

history1 = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE1,
    validation_data=val_generator,    # Use validation split (not test set)
    callbacks=callbacks,
    verbose=1
)

best_phase1 = max(history1.history['val_accuracy'])
print(f"\n✓ Phase 1 best validation accuracy: {best_phase1*100:.1f}%")

# ========================================
# PHASE 2 TRAINING (fine-tune top layers)
# ========================================

print("\n" + "="*70)
print("PHASE 2: Fine-tuning top layers of MobileNetV2")
print("="*70)
print("Unfreezing top 30 layers for fine-tuning. Slower - ~1-3 hours.\n")

# Unfreeze the top 30 layers of MobileNetV2
# Bottom layers detect generic features (edges, colors) - keep frozen
# Top layers detect higher-level features - fine-tune for emotions
base_model.trainable = True
for layer in base_model.layers[:-30]:   # Freeze all except last 30
    layer.trainable = False

# Recompile with a LOWER learning rate
# We use 10x smaller LR to gently adjust pre-trained weights
# without destroying what MobileNetV2 already learned
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE / 10),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE2,
    initial_epoch=EPOCHS_PHASE1,      # Continue epoch count from phase 1
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

best_phase2 = max(history2.history['val_accuracy'])
print(f"\n✓ Phase 2 best validation accuracy: {best_phase2*100:.1f}%")

# Combine histories for plotting
full_history_acc     = history1.history['accuracy']     + history2.history['accuracy']
full_history_val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
full_history_loss    = history1.history['loss']         + history2.history['loss']
full_history_val_loss= history1.history['val_loss']     + history2.history['val_loss']

# Save final model
model.save('my_emotion_model.keras')
print("\n✓ Final model saved as 'my_emotion_model.keras'")

# ========================================
# EVALUATE MODEL PERFORMANCE
# ========================================

print("\n" + "="*70)
print("EVALUATING MODEL")
print("="*70)

# Load the best model (saved by ModelCheckpoint)
best_model = tf.keras.models.load_model('best_emotion_model.h5')

# Calculate accuracy on test set (data the model has NEVER seen)
test_loss, test_acc = best_model.evaluate(test_generator, verbose=0)
print(f"\n✓ Test Accuracy: {test_acc*100:.2f}%")
print(f"✓ Test Loss: {test_loss:.4f}")

# ========================================
# PLOT TRAINING HISTORY
# ========================================

plt.figure(figsize=(15, 5))

# Plot accuracy curves (combined Phase 1 + Phase 2)
plt.subplot(1, 2, 1)
plt.plot(full_history_acc,     label='Training Accuracy',   linewidth=2)
plt.plot(full_history_val_acc, label='Validation Accuracy', linewidth=2)
plt.axvline(x=EPOCHS_PHASE1, color='gray', linestyle='--', label='Fine-tune starts')
plt.title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Plot loss curves
plt.subplot(1, 2, 2)
plt.plot(full_history_loss,     label='Training Loss',   linewidth=2)
plt.plot(full_history_val_loss, label='Validation Loss', linewidth=2)
plt.axvline(x=EPOCHS_PHASE1, color='gray', linestyle='--', label='Fine-tune starts')
plt.title('Model Loss Over Time', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
print("\n✓ Training curves saved as 'training_history.png'")

# ========================================
# GENERATE CONFUSION MATRIX
# ========================================

print("\nGenerating confusion matrix...")

# Get predictions on test set using best model
predictions = best_model.predict(test_generator, verbose=0)
y_pred = np.argmax(predictions, axis=1)  # Get predicted class index for each image
y_true = test_generator.classes           # Get true labels

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred)
emotion_labels = list(train_generator.class_indices.keys())

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=emotion_labels,
            yticklabels=emotion_labels,
            cbar_kws={'label': 'Number of Images'})
plt.title('Confusion Matrix - Emotion Classification', fontsize=16, fontweight='bold')
plt.ylabel('True Emotion', fontsize=12)
plt.xlabel('Predicted Emotion', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print("✓ Confusion matrix saved as 'confusion_matrix.png'")

# ========================================
# DETAILED CLASSIFICATION REPORT
# ========================================

print("\n" + "="*70)
print("CLASSIFICATION REPORT")
print("="*70)
print("\nPer-emotion performance metrics:\n")
print(classification_report(y_true, y_pred, target_names=emotion_labels))

# ========================================
# TRAINING COMPLETE
# ========================================

print("\n" + "="*70)
print(" TRAINING COMPLETE!")
print("="*70)
print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")
print("\nGenerated files:")
print("  • best_emotion_model.h5    - Best model during training (USE THIS)")
print("  • my_emotion_model.keras   - Final model after all epochs")
print("  • training_history.png     - Accuracy/loss curves (Phase 1 + Phase 2)")
print("  • confusion_matrix.png     - Per-emotion performance chart")
print("\nNext step: Run 'python3 facial_analysis_system.py' for real-time detection!")
print("="*70)
