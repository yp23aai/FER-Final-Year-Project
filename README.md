# Facial Expression Recognition System
**Author:** Yodhitomo Sidhi Pranoto  
**SRN:** 23013218  
**University:** University of Hertfordshire

## 📋 Project Overview

This system performs real-time facial analysis using computer vision and deep learning:
- **Emotion Detection:** 7 emotions (Surprise, Fear, Disgust, Happiness, Sadness, Anger, Neutral)
- **Age Estimation:** Approximate age prediction
- **Gender Recognition:** Male/Female classification

## 🎯 Features

### Custom Emotion Model
- Trained on RAF-DB (Real-world Affective Faces Database)
- Custom CNN architecture with 3 convolutional blocks
- Data augmentation for improved generalization
- Achieves competitive accuracy on challenging real-world dataset

### Real-Time Analysis
- Live webcam processing at 10-15 FPS
- Color-coded emotion visualization
- Real-time statistics tracking
- Session analytics with emotion distribution

### User Interface
- Professional information panels
- FPS counter
- Screenshot capability (press 'S')
- Statistics reset function (press 'R')

## 📦 Installation

### 1. Install Required Packages

```bash
# Install dependencies
pip install tensorflow keras opencv-python deepface matplotlib seaborn scikit-learn
```

### 2. Download RAF-DB Dataset

1. Visit: http://www.whdeng.cn/RAF/model1.html
2. Request access to Basic Emotion dataset
3. Download the aligned images and labels
4. Organize as follows:

```
FER_Final_Project/
└── RAF-DB/
    ├── train/
    │   ├── Surprise/
    │   ├── Fear/
    │   ├── Disgust/
    │   ├── Happiness/
    │   ├── Sadness/
    │   ├── Anger/
    │   └── Neutral/
    └── test/
        └── (same structure)
```

## 🚀 Usage

### Step 1: Train Your Emotion Model

```bash
python train_emotion_model.py
```

This will:
- Load RAF-DB dataset with augmentation
- Train custom CNN for 50 epochs (with early stopping)
- Save best model as `my_emotion_model.keras`
- Generate training curves and confusion matrix
- Take 2-4 hours on Mac, GPUs may be faster

### Step 2: Run Real-Time Detection

```bash
python facial_analysis_system.py
```

**Controls:**
- **Q** - Quit application
- **S** - Save screenshot
- **R** - Reset statistics

## 📊 Technical Details

### Model Architecture

```
Input (100x100x3 RGB image)
↓
Block 1: Conv2D(64) → Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)
↓
Block 2: Conv2D(128) → Conv2D(128) → BatchNorm → MaxPool → Dropout(0.25)
↓
Block 3: Conv2D(256) → Conv2D(256) → BatchNorm → MaxPool → Dropout(0.25)
↓
Flatten → Dense(512) → Dropout(0.5) → Dense(256) → Dropout(0.5)
↓
Output: Dense(7, softmax) - 7 emotion classes
```

### Datasets Used

1. **Emotion Detection (Custom Model):**
   - Dataset: RAF-DB (Real-world Affective Faces)
   - Images: ~15,000 training, ~3,000 test
   - Size: 100×100 RGB
   - Classes: 7 basic emotions

2. **Age & Gender (DeepFace):**
   - Age: Trained on IMDB-WIKI (500K+ celebrity images)
   - Gender: VGG-Face architecture
   - Pre-trained models via DeepFace library

3. **Face Detection:**
   - OpenCV Haar Cascade
   - Pre-trained on diverse face datasets

### Data Augmentation

Training uses the following augmentations:
- Random rotation (±20°)
- Horizontal/vertical shifts
- Shearing and zooming
- Horizontal flipping
- Maintains image quality

### Performance

- **Emotion Accuracy:** 65-75% (varies by emotion)
- **Best Performing:** Happiness (~75%)
- **Challenging:** Disgust, Fear (~55-60%)
- **Real-time FPS:** 10-20 FPS on Mac CPU

## 📁 Project Files

```
FER_Final_Project/
├── README.md                      # This file
├── train_emotion_model.py         # Model training script
├── facial_analysis_system.py      # Main software, real-time detection 
├── RAF-DB/                        # Dataset folder 
├── organize_rafdb.py              # Tool to organize RAF-DB dataset
├── best_emotion_model.h5          # ModelCheckpoint best validation accuracy epoch
├── my_emotion_model.keras         # Trained model    
└── requirements.txt               # Python requirements   
```

## 🎓 Academic Integrity

### My Original Work:
- Custom CNN architecture design
- Training script with data augmentation
- Real-time detection system implementation
- User interface and visualization features
- Statistics tracking and analytics
- Integration of multiple components

### External Libraries Used:
- **DeepFace:** Age and gender detection models
  - Serengil, S. I., & Ozpinar, A. (2020). LightFace: A Hybrid Deep Face Recognition Framework.
- **OpenCV:** Face detection (Haar Cascade)
- **TensorFlow/Keras:** Deep learning framework
- **RAF-DB:** Dataset for emotion recognition
  - Li, S., Deng, W., & Du, J. (2017). Reliable crowdsourcing and deep locality-preserving learning for expression recognition in the wild.

## 🆘 Troubleshooting

### Camera doesn't open:
- Check System Preferences → Security & Privacy → Camera
- Grant Terminal/Python access to camera
- Close other apps using camera

### Model fails to load:
- Make sure `train_emotion_model.py` is ran first
- Check that `my_emotion_model.keras` exists
- Verify TensorFlow is installed correctly

### Import errors:
```bash
pip install --upgrade tensorflow keras opencv-python deepface
```

---
